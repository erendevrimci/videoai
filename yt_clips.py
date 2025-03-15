import os
import json
import yt_dlp
from youtube_search import YoutubeSearch
from moviepy.editor import VideoFileClip
import whisper
from datetime import datetime
from scenedetect import VideoManager, SceneManager, open_video
from scenedetect.detectors import ContentDetector
from scenedetect.scene_manager import save_images

def search_youtube(query, max_results=5):
    """Search YouTube for videos matching the query"""
    try:
        results = YoutubeSearch(query, max_results=max_results).to_dict()
        return [{
            'id': video['id'],
            'title': video['title'],
            'url': f"https://www.youtube.com/watch?v={video['id']}",
            'duration': video['duration'],
            'views': video['views'],
            'thumbnail': video['thumbnails'][0]
        } for video in results]
    except Exception as e:
        print(f"Error searching YouTube: {e}")
        return []

def download_video(video_url, output_path="downloads"):
    """Download a YouTube video using yt-dlp"""
    try:
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Extract video ID from URL
        video_id = video_url.split('v=')[-1].split('&')[0]
        filename = f"{video_id}.mp4"
        output_file = os.path.join(output_path, filename)
        
        # Set up yt-dlp options
        ydl_opts = {
            'format': 'best[ext=mp4]',
            'outtmpl': output_file,
            'quiet': True,
            'no_warnings': True,
            'extract_flat': True
        }
        
        # Download the video
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(video_url, download=True)
            
        # Create a simple object with similar properties to pytube's YouTube object
        video_info = {
            'video_id': video_id,
            'length': info.get('duration', 0),
            'author': info.get('uploader', ''),
            'description': info.get('description', ''),
            'title': info.get('title', '')
        }
            
        return output_file, video_info
    except Exception as e:
        print(f"Error downloading video: {e}")
        return None, None

def detect_scenes(video_path, threshold=30.0):
    """Detect scenes in video and return scene list"""
    try:
        video = VideoManager([video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=threshold))
        
        video.start()
        scene_manager.detect_scenes(frame_source=video)
        scene_list = scene_manager.get_scene_list()
        
        video.release()
        return scene_list
    except Exception as e:
        print(f"Error detecting scenes: {e}")
        return []

def split_video_by_scenes(video_path, scene_list):
    """Split video into clips based on detected scenes"""
    try:
        video = VideoFileClip(video_path)
        clips = []
        
        for i, (start, end) in enumerate(scene_list):
            start_time = start.get_seconds()
            end_time = end.get_seconds()
            clip = video.subclip(start_time, end_time)
            clip_path = f"{video_path[:-4]}_scene_{i:03d}.mp4"
            clip.write_videofile(clip_path, codec="libx264", audio_codec="aac", logger=None)
            clips.append({
                'path': clip_path,
                'start_time': start_time,
                'end_time': end_time
            })
        
        video.close()
        return clips
    except Exception as e:
        print(f"Error splitting video: {e}")
        return []

def generate_captions(video_path):
    """Generate captions using Whisper"""
    try:
        model = whisper.load_model("base")
        result = model.transcribe(video_path)
        return result["segments"]
    except Exception as e:
        print(f"Error generating captions: {e}")
        return []

def process_video(query, output_dir="output"):
    """Main function to process a video"""
    # Search for videos
    videos = search_youtube(query, max_results=1)
    if not videos:
        return None
    
    video_info = videos[0]
    
    # Download video
    video_path, video_info_dl = download_video(video_info['url'])
    if not video_path:
        return None
    
    # Detect scenes
    scene_list = detect_scenes(video_path)
    if not scene_list:
        return None
    
    # Split video into clips based on scenes
    clips = split_video_by_scenes(video_path, scene_list)
    
    # Generate metadata
    metadata = {
        'video': {
            'id': video_info['id'],
            'title': video_info['title'],
            'url': video_info['url'],
            'download_date': datetime.now().isoformat(),
            'original_duration': video_info_dl.get('length', 0),
            'author': video_info_dl.get('author', ''),
            'description': video_info_dl.get('description', ''),
            'views': video_info['views'],
            'thumbnail': video_info['thumbnail']
        },
        'clips': []
    }
    
    # Process each clip
    for clip in clips:
        captions = generate_captions(clip['path'])
        clip_info = {
            'path': clip['path'],
            'start_time': clip['start_time'],
            'end_time': clip['end_time'],
            'captions': [{
                'start': segment['start'],
                'end': segment['end'],
                'text': segment['text']
            } for segment in captions]
        }
        metadata['clips'].append(clip_info)
    
    # Save metadata to JSON
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    json_path = os.path.join(output_dir, f"{video_info['id']}_metadata.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)
    
    return json_path

def main():
    # Example usage
    search_query = input("Enter YouTube search query: ")
    try:
        result = process_video(search_query)
        if result:
            print(f"Processing complete. Metadata saved to: {result}")
        else:
            print("Processing failed.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Required dependencies:
    # pip install yt-dlp youtube-search moviepy whisper scenedetect
    main()