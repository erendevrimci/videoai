"""
Captions Generation Module

This module handles the generation of subtitles (SRT format) for videos using OpenAI's Whisper API.
"""

import traceback
from pathlib import Path
from openai import OpenAI
from config import config
from file_manager import FileManager
import json
# from timeline_manager import TimelineManager
import os
from supabase import create_client
from dotenv import load_dotenv
from supabase import Client
from typing import List, Dict, Any
load_dotenv()
# Initialize the file manager
file_mgr = FileManager()

def convert_whisper_to_caption_segments(whisper_json: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    OpenAI Whisper API'den gelen JSON formatını CaptionSegment formatına çevirir.
    
    Args:
        whisper_json: OpenAI Whisper API'den gelen JSON data
        
    Returns:
        List[Dict]: CaptionSegment formatında liste
    """
    try:
        caption_segments = []
        
        # Segments'leri işle
        for segment in whisper_json.get("segments", []):
            segment_id = str(segment.get("id", len(caption_segments) + 1))
            segment_text = segment.get("text", "").strip()
            segment_start = segment.get("start", 0.0)
            segment_end = segment.get("end", 0.0)
            
            # Bu segment için kelime listesini bul
            segment_words = []
            
            # Whisper'daki words array'ini kullan
            if "words" in whisper_json:
                # Segment zamanlamasına göre kelimeleri filtrele
                for word_data in whisper_json["words"]:
                    word_start = word_data.get("start", 0.0)
                    word_end = word_data.get("end", 0.0)
                    
                    # Kelime bu segment'in zaman aralığında mı?
                    if (word_start >= segment_start - 0.1 and 
                        word_end <= segment_end + 0.1):
                        segment_words.append({
                            "word": word_data.get("word", "").strip(),
                            "start": word_start,
                            "end": word_end
                        })
            
            # CaptionSegment formatında segment oluştur
            caption_segment = {
                "id": segment_id,
                "text": segment_text,
                "start": segment_start,
                "end": segment_end,
                "words": segment_words
            }
            
            caption_segments.append(caption_segment)
        
        return caption_segments
        
    except Exception as e:
        print(f"Error converting Whisper JSON to CaptionSegments: {str(e)}")
        return []

def generate_subtitles(
    audio_file: bytes, 
    supabase: Client,
    project_id: int,
    voice_over_id: int,
    channel_number: int = None,
    user_id: str = None
) -> tuple[bool, int]:
    """
    Generate subtitles (SRT file) from the given audio file using the OpenAI Whisper API.
    
    This function sends the generated voice mp3 file to the Whisper transcription API
    and requests the transcription in SRT format. The resulting subtitles are saved 
    into the output file.
    
    Args:
        audio_file_path: The path to the audio file (e.g., a generated_voice.mp3).
        output_srt_path: Path to save the SRT file.
        channel_number: Optional channel number to use for configuration.
        
    Returns:
        bool: True if successful, False otherwise
    """
    
    
    
    # # Check if audio file exists
    # if not Path(audio_file_path).exists():
    #     print(f"Error: Audio file not found at {audio_file_path}")
    #     return False
    
    # Initialize OpenAI client
    client = OpenAI(api_key=config.openai.api_key)
    
    try:
        # # Convert paths to Path objects if they're strings
        # audio_path = Path(audio_file_path) if isinstance(audio_file_path, str) else audio_file_path
        # output_path = Path(output_srt_path) if isinstance(output_srt_path, str) else output_srt_path
        
        # # Ensure the output directory exists using FileManager
        # file_mgr.ensure_dir_exists(output_path.parent)
        
        # print(f"Transcribing audio from {audio_path}...")
        # # Read the audio file using FileManager and process with Whisper API
        audio_data = audio_file
        if audio_data is None:
            print(f"Error: Could not access audio file")
            return False, None
        
        # Create a temporary file for the API to read since it expects a file object
        with file_mgr.temp_file(suffix=".mp3") as temp_audio_path:
            # Write the audio data to the temporary file
            file_mgr.write_binary(temp_audio_path, audio_data)
            
            print("Transcribing audio using OpenAI Whisper API...")
            with open(temp_audio_path, "rb") as audio_file_temp:
                # Request transcription with SRT output format
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",  # Using hardcoded model as Whisper has limited models
                    file=audio_file_temp,
                    response_format="verbose_json",
                    timestamp_granularities=["word","segment"],
                    prompt="each segment should be between 2 to 4 seconds. This means none of the segments should exceed 6 words."
                )
                
                # transcription objesi zaten dict-like bir nesne, to_json() kullanmaya gerek yok
                json_transcription = transcription.model_dump() if hasattr(transcription, 'model_dump') else transcription.__dict__
                
                # CaptionSegment formatına çevir
                caption_segments = convert_whisper_to_caption_segments(json_transcription)
                
                def sec_to_srt(t: float) -> str:
                    h, rem = divmod(t, 3600)
                    m, s = divmod(rem, 60)
                    ms = int((s % 1) * 1000)  # Milisaniye hesapla
                    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

            # SRT formatı için kelime kelime oluştur (daha detaylı timing için)
            srt_lines = []
            idx = 1
            
            # Whisper'daki words array'ini kullanarak kelime kelime SRT oluştur
            for word_data in json_transcription.get("words", []):
                word = word_data.get("word", "").strip()
                start_time = word_data.get("start", 0.0)
                end_time = word_data.get("end", 0.0)
                
                if word:  # Boş kelimeler atla
                    srt_lines.extend([
                        f"{idx}",
                        f"{sec_to_srt(start_time)} --> {sec_to_srt(end_time)}",
                        word,
                        ""
                    ])
                    idx += 1
                    
            srt_content = "\n".join(srt_lines)

            # Segment bazlı SRT içeriği oluştur
            segment_srt_lines = []
            for i, segment in enumerate(caption_segments, 1):
                start = sec_to_srt(segment['start'])
                end = sec_to_srt(segment['end'])
                text = segment['text'].strip()
                segment_srt_lines.append(f"{i}\n{start} --> {end}\n{text}\n")
            
            segment_srt_content = "".join(segment_srt_lines)

        import uuid
        word_srt_filename = f"{uuid.uuid4()}_words.srt"
        segment_srt_filename = f"{uuid.uuid4()}_segments.srt"
        
        # Kelime bazlı SRT'yi yükle
        with file_mgr.temp_file(suffix=".srt") as temp_srt_path:
            with open(temp_srt_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(srt_content)
            
            with open(temp_srt_path, "rb") as srt_file:
                result = supabase.storage.from_("captions").upload(
                    path=word_srt_filename,
                    file=srt_file,
                    file_options={"content-type": "application/x-subrip"}
                )
        if hasattr(result, 'error') and result.error:
            raise Exception(f"Storage Error (word SRT): {result.error}")

        # Segment bazlı SRT'yi yükle
        with file_mgr.temp_file(suffix=".srt") as temp_srt_path:
            with open(temp_srt_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(segment_srt_content)
            
            with open(temp_srt_path, "rb") as srt_file:
                result = supabase.storage.from_("captions").upload(
                    path=segment_srt_filename,
                    file=srt_file,
                    file_options={"content-type": "application/x-subrip"}
                )
        if hasattr(result, 'error') and result.error:
            raise Exception(f"Storage Error (segment SRT): {result.error}")
        
        print(f"Generate User id: {user_id}")
        response = supabase.table("captions").insert({
            "caption_file": word_srt_filename,
            "caption_segment_file": segment_srt_filename,
            "channel_number": channel_number,
            "project_id": project_id,
            "voice_over_id": voice_over_id,
            "caption_json": json.dumps(json_transcription) if isinstance(json_transcription, dict) else json_transcription,
            "caption_segments": json.dumps(caption_segments),
            "segments_count": len(caption_segments),
            "total_duration": json_transcription.get("duration", 0.0),
            "user_id": user_id
        }).execute()

        # Database response kontrolü
        if hasattr(response, 'error') and response.error:
            raise Exception(f"Database Error: {response.error}")
            
        if not response.data or len(response.data) == 0:
            raise Exception("Database insert failed - no data returned")

        caption_id = response.data[0]["id"]
        
        # Hybrid yaklaşım: Kelime bazında veriyi ayrı tabloya da kaydet
        words_saved = save_caption_words_to_table(supabase, caption_id, json_transcription)
        if words_saved:
            print(f"Caption words saved to normalized table for caption {caption_id}")
        
        return True, caption_id
        
    except Exception as e:
        print(f"Error generating subtitles: {str(e)}")
        print(traceback.format_exc())
        return False, None
        

def add_captions_to_timeline(
    captions_path: str | Path,
    channel_number: int = None,
    font: str = "Arial",
    font_size: int = 36, 
    color: str = "#FFFFFF",
    bg_color: str = "#00000080",
    align: str = "center",
    position: str = "bottom"
) -> bool:
    """
    Add captions from an SRT file to a timeline.
    
    Args:
        captions_path: Path to the SRT file
        channel_number: Channel number for context
        font: Font family to use for captions
        font_size: Font size in pixels
        color: Text color in hex format
        bg_color: Background color with alpha in hex format
        align: Text alignment ("left", "center", "right")
        position: Vertical position ("top", "middle", "bottom")
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        print(f"Adding captions from {captions_path} to timeline")
        
        # Initialize TimelineManager
        timeline_mgr = TimelineManager(channel_number=channel_number)
        
        # Check if voice timeline exists
        voice_timeline_path = timeline_mgr.get_timeline_path("voice_timeline")
        if not voice_timeline_path.exists():
            print(f"Voice timeline not found at {voice_timeline_path}")
            # Try to find any existing timeline
            script_timeline_path = timeline_mgr.get_timeline_path("script_timeline")
            if script_timeline_path.exists():
                voice_timeline_path = script_timeline_path
                print(f"Using script timeline instead: {script_timeline_path}")
            else:
                print("No existing timeline found. Creating a new timeline...")
                # Create a new timeline
                timeline = timeline_mgr.create_v3_timeline()
                return timeline_mgr.save_timeline(timeline, "captions_timeline", "Timeline with captions")
        
        # Load the timeline
        timeline = timeline_mgr.load_timeline("voice_timeline")
        if not timeline:
            print(f"Failed to load timeline: {voice_timeline_path}")
            return False
            
        # Add captions to timeline
        timeline = timeline_mgr.add_captions_to_timeline(
            timeline=timeline,
            captions_path=captions_path,
            track_index=1,  # Use track 1 (above the main video)
            font=font,
            font_size=font_size, 
            color=color,
            bg_color=bg_color,
            align=align,
            position=position
        )
        
        # Save updated timeline
        success = timeline_mgr.save_timeline(
            timeline=timeline, 
            timeline_name="captions_timeline",
            description="Timeline with captions added"
        )
        
        print(f"Captions{'successfully' if success else 'failed to be'} added to timeline")
        return success
    
    except Exception as e:
        print(f"Error adding captions to timeline: {str(e)}")
        print(traceback.format_exc())
        return False

def main(project_id: int,voice_over_id: int, channel_number: int = None, use_timeline: bool = False,user_id: str = None):
    """
    Main function to run the captions generation process.
    
    Args:
        channel_number: Optional channel number to use for configuration.
        use_timeline: Whether to use the timeline system for captions.
    """
    # # Use default channel if none specified
    # if channel_number is None:
    #     channel_number = config.default_channel
    
    # # Check for dynamic file paths
    # file_paths_json_path = file_mgr.get_channel_output_path(channel_number) / "current_file_paths.json"
    # dynamic_file_paths = file_mgr.read_json(file_paths_json_path)
    
    # # Use dynamic paths if available
    # if dynamic_file_paths:
    #     if "voice_file" in dynamic_file_paths:
    #         voice_file = dynamic_file_paths["voice_file"]
    #         audio_file_path = file_mgr.get_channel_output_path(channel_number) / voice_file.replace("voice/", "")
    #         print(f"Using dynamic voice file path: {audio_file_path}")
    #     else:
    #         # Fallback to default path
    #         audio_file_path = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","").replace(".mp3",""))
            
    #     if "captions_file" in dynamic_file_paths:
    #         captions_file = dynamic_file_paths["captions_file"] 
    #         output_srt_path = file_mgr.get_channel_output_path(channel_number) / captions_file
    #         print(f"Using dynamic captions file path: {output_srt_path}")
    #     else:
    #         # Fallback to default path
    #         output_srt_path = file_mgr.get_caption_path(channel_number, config.file_paths.captions_file)
    # else:
    #     # Use default paths if no dynamic paths available
    #     audio_file_path = file_mgr.get_audio_output_path(channel_number, config.file_paths.voice_file.replace("voice/","").replace(".mp3",""))
    #     output_srt_path = file_mgr.get_caption_path(channel_number, config.file_paths.captions_file)
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    supabase = create_client(supabase_url, supabase_key)


   
    audio_file_data = supabase.table("voice_over").select("voice_name").eq("project_id", project_id).eq("id", voice_over_id).execute()

    audio_file_name = audio_file_data.data[0]["voice_name"]
    audio_file = supabase.storage.from_("voice-over-files").download(audio_file_name)
    print(f"User id: {user_id}")
    # Generate subtitles
    success, caption_id = generate_subtitles(audio_file, supabase, project_id, voice_over_id, channel_number,user_id)
    
    if success:
        print(f"Caption generation completed successfully for channel {channel_number}")
        return caption_id
        # # If timeline mode enabled, add captions to timeline
        # if use_timeline:
        #     print("Timeline mode enabled. Adding captions to timeline...")
        #     srt_file_query = supabase.table("captions").select("caption_file").eq("voice_over_id", voiceover_id).execute()
        #     srt_file_name = srt_file_query.data[0]["caption_file"]
        #     srt_file = supabase.storage.from_("captions").download(srt_file_name)
            
        #     # İndirilen SRT dosyasını geçici bir dosyaya kaydet
        #     with file_mgr.temp_file(suffix=".srt") as temp_srt_path:
        #         # Metin içeriğini dosyaya yaz
        #         with open(temp_srt_path, "wb") as f:
        #             f.write(srt_file)
                
        #         # Şimdi geçici dosya yolunu kullanarak altyazıları timeline'a ekle
        #         timeline_success = add_captions_to_timeline(
        #             captions_path=temp_srt_path,
        #             channel_number=channel_number
        #         )
            
        #     if timeline_success:
        #         print("Captions successfully added to timeline")
                
        #         # Visualize the timeline to show the captions
        #         timeline_mgr = TimelineManager(channel_number=channel_number)
        #         timeline = timeline_mgr.load_timeline("captions_timeline")
        #         if timeline:
        #             timeline_mgr.export_timeline_visualization(
        #                 timeline=timeline,
        #                 detail_level="normal"
        #             )
        #             print("Timeline visualization exported")
        #     else:
        #         print("Failed to add captions to timeline")
    else:
        print(f"Caption generation failed for channel {channel_number}")
        return None

def get_caption_segments(supabase: Client, caption_id: int) -> List[Dict[str, Any]]:
    """
    Veritabanından belirli bir caption_id için CaptionSegment formatındaki veriyi getirir.
    
    Args:
        supabase: Supabase client
        caption_id: Caption ID
        
    Returns:
        List[Dict]: CaptionSegment formatında liste
    """
    try:
        response = supabase.table("captions").select("caption_segments").eq("id", caption_id).execute()
        
        if response.data and len(response.data) > 0:
            caption_segments_json = response.data[0]["caption_segments"]
            return json.loads(caption_segments_json) if isinstance(caption_segments_json, str) else caption_segments_json
        
        return []
        
    except Exception as e:
        print(f"Error fetching caption segments: {str(e)}")
        return []

def get_captions_by_project(supabase: Client, project_id: int) -> List[Dict[str, Any]]:
    """
    Belirli bir proje için tüm caption'ları CaptionSegment formatında getirir.
    
    Args:
        supabase: Supabase client
        project_id: Project ID
        
    Returns:
        List[Dict]: Her caption için metadata ve segments
    """
    try:
        response = supabase.table("captions").select(
            "id, caption_file, caption_segments, segments_count, total_duration, created_at"
        ).eq("project_id", project_id).execute()
        
        captions = []
        for caption_data in response.data:
            caption_segments = json.loads(caption_data["caption_segments"]) if isinstance(caption_data["caption_segments"], str) else caption_data["caption_segments"]
            
            captions.append({
                "id": caption_data["id"],
                "caption_file": caption_data["caption_file"],
                "segments": caption_segments,
                "segments_count": caption_data["segments_count"],
                "total_duration": caption_data["total_duration"],
                "created_at": caption_data["created_at"]
            })
            
        return captions
        
    except Exception as e:
        print(f"Error fetching captions by project: {str(e)}")
        return []

def save_caption_words_to_table(supabase: Client, caption_id: int, whisper_json: Dict[str, Any]) -> bool:
    """
    Caption kelimelerini ayrı bir tabloya kaydeder (hybrid yaklaşım).
    Hem JSON'da hem de normalized tabloda tutar.
    
    Args:
        supabase: Supabase client
        caption_id: Caption ID
        whisper_json: Whisper API response
        
    Returns:
        bool: Success status
    """
    try:
        words_data = []
        
        for idx, word_data in enumerate(whisper_json.get("words", [])):
            words_data.append({
                "caption_id": caption_id,
                "word_index": idx,
                "word": word_data.get("word", "").strip(),
                "start_time": word_data.get("start", 0.0),
                "end_time": word_data.get("end", 0.0),
                "confidence": 1.0  # Whisper API confidence değeri varsa buraya eklenebilir
            })
        
        # Batch insert
        if words_data:
            response = supabase.table("caption_words").insert(words_data).execute()
            
            if hasattr(response, 'error') and response.error:
                raise Exception(f"Error inserting caption words: {response.error}")
                
            print(f"Inserted {len(words_data)} words for caption {caption_id}")
            return True
            
        return False
        
    except Exception as e:
        print(f"Error saving caption words: {str(e)}")
        return False

def get_editable_caption_words(supabase: Client, caption_id: int) -> List[Dict[str, Any]]:
    """
    Caption editörü için kelime bazında veri getirir.
    
    Args:
        supabase: Supabase client
        caption_id: Caption ID
        
    Returns:
        List[Dict]: Editable word format
    """
    try:
        response = supabase.table("caption_words").select(
            "id, word_index, word, start_time, end_time, confidence"
        ).eq("caption_id", caption_id).order("word_index").execute()
        
        return response.data if response.data else []
        
    except Exception as e:
        print(f"Error fetching editable caption words: {str(e)}")
        return []

def update_caption_word_timing(supabase: Client, word_id: int, start_time: float, end_time: float) -> bool:
    """
    Belirli bir kelimenin timing'ini günceller.
    
    Args:
        supabase: Supabase client
        word_id: Word ID
        start_time: New start time
        end_time: New end time
        
    Returns:
        bool: Success status
    """
    try:
        response = supabase.table("caption_words").update({
            "start_time": start_time,
            "end_time": end_time
        }).eq("id", word_id).execute()
        
        return not (hasattr(response, 'error') and response.error)
        
    except Exception as e:
        print(f"Error updating caption word timing: {str(e)}")
        return False

def format_caption_for_api(supabase: Client, caption_id: int) -> Dict[str, Any]:
    """
    Caption verilerini API response formatına çevirir.
    
    Args:
        supabase: Supabase client
        caption_id: Caption ID
        
    Returns:
        Dict: API response formatında caption verisi
    """
    try:
        # Caption ana verilerini al
        response = supabase.table("captions").select(
            "id, project_id, voice_over_id, caption_file, caption_segments, total_duration, created_at"
        ).eq("id", caption_id).execute()
        
        if not response.data or len(response.data) == 0:
            return None
            
        caption_data = response.data[0]
        
        # Segments'leri parse et
        caption_segments = caption_data.get("caption_segments", "[]")
        if isinstance(caption_segments, str):
            segments = json.loads(caption_segments)
        else:
            segments = caption_segments
            
        # API formatına dönüştür
        formatted_caption = {
            "id": caption_data["id"],
            "project_id": caption_data["project_id"],
            "voice_over_id": caption_data["voice_over_id"],
            "caption_file": caption_data["caption_file"],
            "segments": segments,
            "total_duration": caption_data.get("total_duration", 0.0),
            "created_at": caption_data["created_at"]
        }
        
        return formatted_caption
        
    except Exception as e:
        print(f"Error formatting caption for API: {str(e)}")
        return None

def update_caption_segments_in_db(supabase: Client, caption_id: int, segments: List[Dict[str, Any]]) -> bool:
    """
    Caption segments'lerini veritabanında günceller.
    
    Args:
        supabase: Supabase client
        caption_id: Caption ID
        segments: Güncellenmiş segment listesi
        
    Returns:
        bool: Success status
    """
    try:
        # Segments'leri JSON string'e çevir
        segments_json = json.dumps(segments)
        
        # Veritabanını güncelle
        response = supabase.table("captions").update({
            "caption_segments": segments_json,
            "segments_count": len(segments),
            "updated_at": "NOW()"
        }).eq("id", caption_id).execute()
        
        if hasattr(response, 'error') and response.error:
            print(f"Error updating caption segments: {response.error}")
            return False
            
        # Normalized table'ı da güncelle (varsa)
        try:
            # Önce mevcut kelimeleri sil
            supabase.table("caption_words").delete().eq("caption_id", caption_id).execute()
            
            # Yeni kelimeleri ekle
            words_data = []
            for segment in segments:
                for idx, word_data in enumerate(segment.get("words", [])):
                    words_data.append({
                        "caption_id": caption_id,
                        "word_index": idx,
                        "word": word_data.get("word", ""),
                        "start_time": word_data.get("start", 0.0),
                        "end_time": word_data.get("end", 0.0),
                        "confidence": 1.0
                    })
            
            if words_data:
                supabase.table("caption_words").insert(words_data).execute()
                
        except Exception as word_update_error:
            print(f"Warning: Failed to update normalized caption_words: {str(word_update_error)}")
            # Ana güncelleme başarılı olduğu için True döndür
            
        return True
        
    except Exception as e:
        print(f"Error updating caption segments: {str(e)}")
        return False

def get_captions_by_project_for_api(supabase: Client, project_id: int) -> List[Dict[str, Any]]:
    """
    Proje için tüm caption'ları API formatında getirir.
    
    Args:
        supabase: Supabase client
        project_id: Project ID
        
    Returns:
        List[Dict]: API formatında caption listesi
    """
    try:
        response = supabase.table("captions").select(
            "id, project_id, voice_over_id, caption_file, caption_segments, total_duration, created_at"
        ).eq("project_id", project_id).order("created_at", desc=True).execute()
        
        captions = []
        for caption_data in response.data:
            formatted_caption = format_caption_for_api(supabase, caption_data["id"])
            if formatted_caption:
                captions.append(formatted_caption)
                
        return captions
        
    except Exception as e:
        print(f"Error fetching captions by project: {str(e)}")
        return []

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate captions for video using OpenAI Whisper API")
    parser.add_argument("--channel", type=int, help="Channel number to use for configuration")
    parser.add_argument("--timeline", action="store_true", help="Enable timeline mode to add captions to timeline")
    parser.add_argument("--font", type=str, default="Arial", help="Font family to use for captions")
    parser.add_argument("--font-size", type=int, default=36, help="Font size in pixels")
    parser.add_argument("--color", type=str, default="#FFFFFF", help="Text color in hex format")
    parser.add_argument("--bg-color", type=str, default="#00000080", help="Background color with alpha in hex format")
    parser.add_argument("--position", type=str, default="bottom", choices=["top", "middle", "bottom"], help="Vertical position")
    
    args = parser.parse_args()
    
    # Call main with timeline flag
    main(channel_number=args.channel, use_timeline=args.timeline)
