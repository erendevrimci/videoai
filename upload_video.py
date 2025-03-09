"""
YouTube Video Upload Module

This module handles the authentication, video uploading, and thumbnail selection for 
YouTube channels using the YouTube Data API.
"""

import os
import json
import random
import pickle
import traceback
from pathlib import Path
from typing import Dict, Any, Optional, List, Union

import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
import google.generativeai as genai

from config import config, get_channel_config
from file_manager import FileManager

# Initialize the file manager
file_mgr = FileManager()

# YouTube API scopes required for uploading videos
SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
CLIENT_SECRETS_FILE = "client_secret.json"  # Google OAuth client secrets

# Fix for redirect_uri_mismatch error
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'  # For local development only


def initialize_gemini_client():
    """Initialize the Gemini AI client with API key from config"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API key not found in environment variables")
    
    genai.configure(api_key=gemini_api_key)


def authenticate(channel_id: str) -> googleapiclient.discovery.Resource:
    """
    Authenticates and authorizes access to the YouTube Data API.
    Saves and loads credentials from token.pickle file.
    
    Args:
        channel_id: Identifier for the channel to authenticate
        
    Returns:
        An authenticated YouTube API client
    """
    credentials = None
    token_file = f"token_{channel_id}.pickle"
    
    # Load the saved credentials from token_{channel_id}.pickle if it exists
    if os.path.exists(token_file):
        print(f"Loading credentials for channel {channel_id} from file...")
        with open(token_file, 'rb') as token:
            credentials = pickle.load(token)
    
    # If there are no valid credentials available, authenticate
    if not credentials or not credentials.valid:
        if credentials and credentials.expired and credentials.refresh_token:
            print("Refreshing access token...")
            credentials.refresh(Request())
        else:
            print(f"Fetching new tokens for channel {channel_id}...")
            
            # Check if client secrets file exists
            if not os.path.exists(CLIENT_SECRETS_FILE):
                raise FileNotFoundError(f"{CLIENT_SECRETS_FILE} not found. Please download it from Google API Console.")
                
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            # Use a consistent port number and consistent redirect URI
            # This must match exactly what's configured in Google Cloud Console
            print("Opening browser for authentication, please log in...")
            credentials = flow.run_local_server(port=8080, 
                                              success_message="Authentication successful! You can close this window now.",
                                              open_browser=True)
            
        # Save the credentials for the next run
        print(f"Saving credentials for channel {channel_id}...")
        with open(token_file, 'wb') as token:
            pickle.dump(credentials, token)
    
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)


def load_youtube_info(info_file: Path) -> Dict[str, Any]:
    """
    Load title and description from YouTube info JSON file
    
    Args:
        info_file: Path to the YouTube info JSON file
        
    Returns:
        Dictionary containing title, description, and tags
    """
    # Use FileManager to read JSON
    data = file_mgr.read_json(info_file)
    
    if data is None:
        print(f"Error loading YouTube info from {info_file}")
        return {
            "title": "Error: Title not available",
            "description": "Error: Description not available",
            "tags": ["ai", "technology"]
        }
    
    return data


def upload_video(
    youtube: googleapiclient.discovery.Resource, 
    video_file_path: Path, 
    youtube_info: Dict[str, Any],
    category_id: str = "28",
    privacy_status: str = "private"
) -> Optional[str]:
    """
    Uploads a video to YouTube.
    
    Args:
        youtube: Authenticated YouTube API client
        video_file_path: Path to the video file to upload
        youtube_info: Dictionary containing title, description, and optional tags
        category_id: YouTube category ID (default is "28" for Science & Technology)
        privacy_status: Privacy status of the video (public, private, unlisted)
        
    Returns:
        Video ID if upload succeeds, None if it fails
    """
    try:
        # Prepare tags from youtube_info or default
        tags = youtube_info.get("tags", ["ai", "openai", "llms", "robotics", "future", "tech"])
        
        # Prepare the request body
        body = {
            "snippet": {
                "title": youtube_info["title"],
                "description": youtube_info["description"],
                "tags": tags,
                "categoryId": category_id
            },
            "status": {
                "privacyStatus": privacy_status  # Can be "public", "private", or "unlisted"
            }
        }

        # Create MediaFileUpload object
        media_file = MediaFileUpload(
            video_file_path, 
            mimetype='video/mp4',
            resumable=True
        )

        print(f"Uploading video: {youtube_info['title']}")
        print(f"Privacy status: {privacy_status}")
        
        # Create the video insert request
        request = youtube.videos().insert(
            part=",".join(body.keys()),
            body=body,
            media_body=media_file
        )

        # Upload the video
        response = None
        while response is None:
            status, response = request.next_chunk()
            if status:
                print(f"Uploaded {int(status.progress() * 100)}%.")
        
        video_id = response['id']
        print(f"Video id '{video_id}' was successfully uploaded.")
        return video_id
        
    except Exception as e:
        print(f"Error uploading video: {str(e)}")
        print(traceback.format_exc())
        return None


def upload_thumbnail(
    youtube: googleapiclient.discovery.Resource, 
    video_id: str, 
    thumbnail_path: Path
) -> bool:
    """
    Uploads a thumbnail for a specific video.
    
    Args:
        youtube: Authenticated YouTube API client
        video_id: ID of the uploaded video
        thumbnail_path: Path to the thumbnail image
        
    Returns:
        True if successful, False otherwise
    """
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path)
        ).execute()
        print(f"Thumbnail successfully uploaded for video {video_id}")
        return True
    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while uploading thumbnail: {e.content}")
        return False
    except Exception as e:
        print(f"Error uploading thumbnail: {str(e)}")
        print(traceback.format_exc())
        return False


def read_file_content(file_path: Path) -> str:
    """
    Read and return the content of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File content as string
    """
    # Use FileManager to read text file
    content = file_mgr.read_text(file_path)
    return content if content is not None else ""


def read_last_thumbnails(base_dir: Path = None) -> List[str]:
    """
    Read the last used thumbnails from JSON file.
    
    Args:
        base_dir: Base directory for the last_thumb.json file
        
    Returns:
        List of recently used thumbnail filenames
    """
    last_thumb_file = Path(base_dir) / "last_thumb.json" if base_dir else Path("last_thumb.json")
    
    # Use FileManager to read JSON
    data = file_mgr.read_json(last_thumb_file)
    
    if data is None:
        return []
    
    return data.get("last_thumbnails", [])


def update_last_thumbnails(selected_thumbnail: str, base_dir: Path = None, max_history: int = 6) -> None:
    """
    Update the list of last used thumbnails, keeping only the most recent ones.
    
    Args:
        selected_thumbnail: Filename of the selected thumbnail
        base_dir: Base directory for the last_thumb.json file
        max_history: Maximum number of thumbnails to keep in history
    """
    last_thumb_file = Path(base_dir) / "last_thumb.json" if base_dir else Path("last_thumb.json")
    
    # Get current thumbnails
    thumbnails = read_last_thumbnails(base_dir)
    
    # Add new thumbnail to the list
    thumbnails.append(selected_thumbnail)
    
    # Keep only the last max_history thumbnails
    if len(thumbnails) > max_history:
        thumbnails = thumbnails[-max_history:]
    
    # Save updated list using FileManager
    data = {"last_thumbnails": thumbnails}
    success = file_mgr.write_json(last_thumb_file, data)
    
    if not success:
        print(f"Error updating thumbnail history in {last_thumb_file}")


def select_best_thumbnail(
    youtube_info: Dict[str, Any], 
    thumbnail_labels: str, 
    thumbnail_dir: Path, 
    base_dir: Path = None
) -> Optional[Path]:
    """
    Use Gemini AI to select the best thumbnail based on video title and description.
    
    Args:
        youtube_info: Dictionary containing video title and description
        thumbnail_labels: Markdown content describing each thumbnail
        thumbnail_dir: Directory containing thumbnail images
        base_dir: Base directory for the last_thumb.json file
        
    Returns:
        Path to the selected thumbnail, or None if selection failed
    """
    try:
        # Initialize Gemini client
        initialize_gemini_client()
        
        # Get last used thumbnails
        last_used = read_last_thumbnails(base_dir)
        
        prompt = f"""
        Given this video title and description:
        Title: {youtube_info["title"]}
        Description: {youtube_info["description"]}
        
        And these available thumbnails with their descriptions:
        {thumbnail_labels}
        
        Please avoid these recently used thumbnails:
        {', '.join(last_used)}
        
        Which thumbnail would be the most appropriate for this video content?
        If the video is about Google AI, use a google related thumbnail.
        If the video is about OpenAI, use an openai related thumbnail.
        If the video is about Meta, use a meta related thumbnail.
        If the video is about robotics, use a robotics related thumbnail.
        And so and so forth.
        
        Return only the image filename, nothing else (VERY IMPORTANT)
        """
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(
            prompt,
            generation_config=genai.GenerationConfig(
                temperature=0.1,
                max_output_tokens=500
            )
        )
        
        selected_thumbnail = response.text.strip()
        thumbnail_path = thumbnail_dir / selected_thumbnail
        
        # Update history if thumbnail exists
        if thumbnail_path.exists():
            update_last_thumbnails(selected_thumbnail, base_dir)
            return thumbnail_path
        else:
            print(f"Selected thumbnail {selected_thumbnail} not found in thumbnails folder")
            return None
        
    except Exception as e:
        print(f"Error selecting thumbnail with AI: {str(e)}")
        print(traceback.format_exc())
        return None


def handle_thumbnail_upload(
    youtube: googleapiclient.discovery.Resource, 
    video_id: str, 
    youtube_info: Dict[str, Any], 
    base_dir: Path = None
) -> bool:
    """
    Handle thumbnail selection and upload for a specific channel
    
    Args:
        youtube: Authenticated YouTube API client
        video_id: ID of the uploaded video
        youtube_info: Dictionary containing video title and description
        base_dir: Base directory for assets
        
    Returns:
        True if thumbnail was uploaded, False otherwise
    """
    # Set thumbnail directory path
    base_dir = base_dir or Path(".")
    thumbnail_dir = base_dir / "thumbnails"
    
    if thumbnail_dir.exists():
        thumbnail_labels_file = thumbnail_dir / "thumbnail_labels.md"
        
        if thumbnail_labels_file.exists():
            thumbnail_labels = read_file_content(thumbnail_labels_file)
            
            try:
                # Try AI-based selection
                thumbnail_path = select_best_thumbnail(
                    youtube_info, 
                    thumbnail_labels, 
                    thumbnail_dir, 
                    base_dir
                )
                
                if thumbnail_path and thumbnail_path.exists():
                    if upload_thumbnail(youtube, video_id, thumbnail_path):
                        print(f"Selected thumbnail based on content analysis: {thumbnail_path.name}")
                        return True
                
                # If AI selection fails, fall back to random selection
                print("Falling back to random selection...")
                thumbnail_files = list(thumbnail_dir.glob("*.jpg")) + list(thumbnail_dir.glob("*.png"))
                
                if thumbnail_files:
                    random_thumbnail = random.choice(thumbnail_files)
                    if upload_thumbnail(youtube, video_id, random_thumbnail):
                        print(f"Selected thumbnail (random fallback): {random_thumbnail.name}")
                        update_last_thumbnails(random_thumbnail.name, base_dir)
                        return True
            except Exception as e:
                print(f"Error in thumbnail selection process: {str(e)}")
                print(traceback.format_exc())
        else:
            print(f"Thumbnail labels file not found: {thumbnail_labels_file}")
    else:
        print(f"Thumbnail directory not found: {thumbnail_dir}")
    
    return False


def upload_to_channel(channel_number: int) -> Optional[str]:
    """
    Upload video to a specific channel
    
    Args:
        channel_number: Channel number to upload to
        
    Returns:
        Video ID if successful, None otherwise
    """
    try:
        # Get channel configuration
        channel_config = get_channel_config(channel_number)
        channel_id = f"channel{channel_number}"
        
        # Get file paths using FileManager
        channel_output_dir = file_mgr.get_channel_output_path(channel_number)
        video_file_path = file_mgr.get_video_output_path(channel_number, "final_output_with_subtitles")
        youtube_info_file = channel_output_dir / channel_config.youtube_info_file
        
        print(f"\n=== Uploading video for channel {channel_number} ({channel_config.name}) ===")
        print(f"Video file: {video_file_path}")
        print(f"YouTube info file: {youtube_info_file}")
        
        # Verify files exist using FileManager
        if not file_mgr.file_exists(video_file_path):
            print(f"Error: Video file not found at {video_file_path}")
            print("Creating emergency placeholder video for upload...")
            
            try:
                # Import video_edit only when needed to avoid circular imports
                from video_edit import create_placeholder_clip
                
                # Create a placeholder video
                create_placeholder_clip(video_file_path, 60)
                print(f"Created placeholder video at {video_file_path}")
                
                # Verify the placeholder was created
                if not file_mgr.file_exists(video_file_path):
                    print(f"Failed to create placeholder video at {video_file_path}")
                    return None
            except Exception as e:
                print(f"Error creating placeholder video: {e}")
                return None
            
        if not file_mgr.file_exists(youtube_info_file):
            print(f"Error: YouTube info file not found at {youtube_info_file}")
            return None
        
        # Authenticate with YouTube
        youtube = authenticate(channel_id)
        
        # Load video info using FileManager
        youtube_info = load_youtube_info(youtube_info_file)
        
        # Upload video
        video_id = upload_video(
            youtube, 
            video_file_path, 
            youtube_info, 
            category_id=config.youtube.category_id,
            privacy_status=channel_config.privacy_status
        )
        
        if video_id:
            # Handle thumbnail upload
            handle_thumbnail_upload(youtube, video_id, youtube_info, base_dir=file_mgr.base_dir)
            return video_id
        
        return None
    
    except Exception as e:
        print(f"Error uploading to channel {channel_number}: {str(e)}")
        print(traceback.format_exc())
        return None


def main(channel_numbers: Union[int, List[int], None] = None):
    """
    Main function to upload videos to specified channels
    
    Args:
        channel_numbers: List of channel numbers to upload to, single channel number,
                        or None to use all channels
    """
    # Handle case where a single integer is passed (from main.py)
    if isinstance(channel_numbers, int):
        channel_numbers = [channel_numbers]
    
    # If no channels specified, use all configured channels
    if not channel_numbers:
        channel_numbers = list(config.channels.keys())
    
    print(f"Starting video upload process for channels: {channel_numbers}")
    
    # Upload to each channel
    for channel_num in channel_numbers:
        video_id = upload_to_channel(channel_num)
        if video_id:
            print(f"Successfully uploaded video to channel {channel_num}. Video ID: {video_id}")
        else:
            print(f"Failed to upload video to channel {channel_num}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload video to YouTube channels")
    parser.add_argument("--channel", type=int, help="Specific channel number to upload to")
    parser.add_argument("--channels", type=int, nargs='+', help="List of channel numbers to upload to")
    
    args = parser.parse_args()
    
    if args.channel:
        # Single channel specified
        main([args.channel])
    elif args.channels:
        # Multiple channels specified
        main(args.channels)
    else:
        # No channels specified, use all
        main()