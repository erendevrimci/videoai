"""
Captions Generation Module

This module handles the generation of subtitles (SRT format) for videos using OpenAI's Whisper API.
"""

import traceback
from pathlib import Path
from openai import OpenAI
from config import config
from file_manager import FileManager
# from timeline_manager import TimelineManager
import os
from supabase import create_client
from dotenv import load_dotenv
from supabase import Client
load_dotenv()
# Initialize the file manager
file_mgr = FileManager()

def generate_subtitles(
    audio_file: bytes, 
    supabase: Client,
    user_id: str,
    voice_over_id: int,
    channel_number: int = None
) -> bool:
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
            return False
        
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
                    response_format="srt"
                )
        import uuid
        file_name = f"{uuid.uuid4()}.srt"
        
        # Transkripsiyon içeriğini geçici bir dosyaya yaz
        with file_mgr.temp_file(suffix=".srt") as temp_srt_path:
            # Metin içeriğini dosyaya yaz
            with open(temp_srt_path, "w", encoding="utf-8") as srt_file:
                srt_file.write(transcription)
            
            # Dosyayı Supabase'e yükle
            with open(temp_srt_path, "rb") as srt_file:
                result = supabase.storage.from_("captions").upload(
                    path=file_name,
                    file=srt_file,
                    file_options={"content-type": "application/x-subrip"}
                )
        
        
        try:
            
            error = getattr(result, "error", None)
            if error:
                raise Exception(f"Storage Error: {error}")
        except Exception as e:
            if "error" in str(e).lower():
                raise Exception(f"Storage Error: {str(e)}")
            
        response = supabase.table("captions").insert({
            "user_id": user_id,
            "voice_over_id": voice_over_id,
            "caption_file": file_name,
            "channel_number": channel_number
        }).execute()
        if "error" in response:
            raise Exception(f"Database Error: {response}")
        return True
    except Exception as e:
        print(f"Error generating subtitles: {str(e)}")
        print(traceback.format_exc())
        return False
        

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

def main(voiceover_id: int, user_id: str, channel_number: int = None, use_timeline: bool = False):
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
    audio_file_data = supabase.table("voice_over").select("voice_name").eq("id", voiceover_id).execute()
    print(audio_file_data)
    audio_file_name = audio_file_data.data[0]["voice_name"]
    audio_file = supabase.storage.from_("voice-over-files").download(audio_file_name)
    # Generate subtitles
    success = generate_subtitles(audio_file, supabase, user_id, voiceover_id, channel_number)
    
    if success:
        print(f"Caption generation completed successfully for channel {channel_number}")
        
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
