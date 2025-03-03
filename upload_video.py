import google_auth_oauthlib.flow
import googleapiclient.discovery
import googleapiclient.errors
from googleapiclient.http import MediaFileUpload
from google.oauth2.credentials import Credentials
import json
import os
import random
import pickle
import google.generativeai as genai
from dotenv import load_dotenv
import requests
from google.auth.transport.requests import Request

load_dotenv(override=True)

# Configure Gemini API
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

SCOPES = ["https://www.googleapis.com/auth/youtube.upload", "https://www.googleapis.com/auth/youtube"]
CLIENT_SECRETS_FILE = "client_secret.json"
TOKEN_PICKLE_FILE = "token.pickle"

VIDEO_FILE_PATH = "final_output_with_subtitles.mp4"  # Replace with the path to your video
CATEGORY_ID = "28"  # See list below
MODEL_NAME = "gemini-2.0-flash"


def authenticate(channel_id="default"):
    """Authenticates and authorizes access to the YouTube Data API.
    Saves and loads credentials from token.pickle file.
    
    Args:
        channel_id (str): Identifier for the channel to authenticate
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
            flow = google_auth_oauthlib.flow.InstalledAppFlow.from_client_secrets_file(
                CLIENT_SECRETS_FILE, SCOPES)
            credentials = flow.run_local_server(port=0)
            
        # Save the credentials for the next run
        print(f"Saving credentials for channel {channel_id}...")
        with open(token_file, 'wb') as token:
            pickle.dump(credentials, token)
    
    return googleapiclient.discovery.build("youtube", "v3", credentials=credentials)

def load_youtube_info():
    """Load title and description from youtube_info.json"""
    with open("youtube_info.json", "r", encoding="utf-8") as f:
        return json.load(f)

def upload_video(youtube, video_file_path):
    """Uploads a video to YouTube."""
    # Load title and description from JSON
    youtube_info = load_youtube_info()
    
    body = {
        "snippet": {
            "title": youtube_info["title"],
            "description": youtube_info["description"],
            "tags": ["ai", "openai", "llms", "robotics", "future", "tech"],
            "categoryId": CATEGORY_ID
        },
        "status": {
            "privacyStatus": "private"  # Can be "public", "private", or "unlisted"
        }
    }

    # Create MediaFileUpload object
    media_file = MediaFileUpload(video_file_path, 
                                mimetype='video/mp4',
                                resumable=True)

    # Create the video insert request
    request = youtube.videos().insert(
        part=",".join(body.keys()),
        body=body,
        # The media_body parameter is used instead of media
        media_body=media_file
    )

    # Upload the video
    response = None
    while response is None:
        status, response = request.next_chunk()
        if status:
            print(f"Uploaded {int(status.progress() * 100)}%.")
    
    print(f"Video id '{response['id']}' was successfully uploaded.")
    return response['id']  # Return video ID for thumbnail upload

def upload_thumbnail(youtube, video_id, thumbnail_path):
    """Uploads a thumbnail for a specific video."""
    try:
        youtube.thumbnails().set(
            videoId=video_id,
            media_body=MediaFileUpload(thumbnail_path)
        ).execute()
        print(f"Thumbnail successfully uploaded for video {video_id}")
    except googleapiclient.errors.HttpError as e:
        print(f"An HTTP error {e.resp.status} occurred while uploading thumbnail: {e.content}")

def read_file_content(file_path):
    """Read and return the content of a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

def read_last_thumbnails():
    """Read the last used thumbnails from JSON file."""
    try:
        with open("last_thumb.json", "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {"last_thumbnails": []}

def update_last_thumbnails(selected_thumbnail):
    """Update the list of last used thumbnails, keeping only the last 6."""
    data = read_last_thumbnails()
    thumbnails = data["last_thumbnails"]
    
    # Add new thumbnail to the list
    thumbnails.append(selected_thumbnail)
    
    # Keep only the last 6 thumbnails
    if len(thumbnails) > 6:
        thumbnails = thumbnails[-6:]
    
    # Save updated list
    with open("last_thumb.json", "w") as f:
        json.dump({"last_thumbnails": thumbnails}, f)

def select_best_thumbnail(youtube_info, thumbnail_labels):
    """Use Gemini AI to select the best thumbnail based on video title and description."""
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Get last used thumbnails
    last_used = read_last_thumbnails()["last_thumbnails"]
    
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
    
    model = genai.GenerativeModel(MODEL_NAME)
    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.1,
            max_output_tokens=500
        )
    )
    
    selected_thumbnail = response.text.strip()
    update_last_thumbnails(selected_thumbnail)
    return selected_thumbnail

def upload_to_channel1():
    """Upload video to channel 1"""
    youtube = authenticate("channel1")
    video_id = upload_video(youtube, VIDEO_FILE_PATH)
    
    # Handle thumbnail upload for channel 1
    handle_thumbnail_upload(youtube, video_id)

def upload_to_channel2():
    """Upload video to channel 2"""
    youtube = authenticate("channel2")
    video_id = upload_video(youtube, VIDEO_FILE_PATH)
    
    # Handle thumbnail upload for channel 2
    handle_thumbnail_upload(youtube, video_id)

def upload_to_channel3():
    """Upload video to channel 3"""
    youtube = authenticate("channel3")
    video_id = upload_video(youtube, VIDEO_FILE_PATH)
    
    # Handle thumbnail upload for channel 3
    handle_thumbnail_upload(youtube, video_id)

def handle_thumbnail_upload(youtube, video_id):
    """Handle thumbnail selection and upload for a specific channel"""
    thumbnail_folder = "thumbnails"
    if os.path.exists(thumbnail_folder):
        youtube_info = load_youtube_info()
        thumbnail_labels = read_file_content("thumbnails/thumbnail_labels.md")
        
        try:
            selected_thumbnail = select_best_thumbnail(youtube_info, thumbnail_labels)
            thumbnail_path = os.path.join(thumbnail_folder, selected_thumbnail)
            
            if os.path.exists(thumbnail_path):
                upload_thumbnail(youtube, video_id, thumbnail_path)
                print(f"Selected thumbnail based on content analysis: {selected_thumbnail}")
            else:
                print(f"Selected thumbnail {selected_thumbnail} not found in thumbnails folder")
        except Exception as e:
            print(f"Error selecting thumbnail with AI: {str(e)}")
            print("Falling back to random selection...")
            thumbnail_files = [f for f in os.listdir(thumbnail_folder) 
                             if f.lower().endswith(('.jpg', '.png'))]
            if thumbnail_files:
                random_thumbnail = os.path.join(thumbnail_folder, random.choice(thumbnail_files))
                upload_thumbnail(youtube, video_id, random_thumbnail)

def main():
    # Example of authenticating multiple channels
    channel1 = authenticate("channel1")  # This will create token_channel1.pickle
    channel2 = authenticate("channel2")  # This will create token_channel2.pickle
    channel3 = authenticate("channel3")  # This will create token_channel3.pickle
    
    # You can now use different channel instances for uploads
    video_id = upload_video(channel1, VIDEO_FILE_PATH)  # Upload to channel 1
    video_id2 = upload_video(channel2, VIDEO_FILE_PATH)  # Upload to channel 2
    video_id3 = upload_video(channel3, VIDEO_FILE_PATH)  # Upload to channel 3
    
    # Continue with thumbnail upload for the specific channel
    thumbnail_folder = "thumbnails"
    if os.path.exists(thumbnail_folder):
        youtube_info = load_youtube_info()
        thumbnail_labels = read_file_content("thumbnails/thumbnail_labels.md")
        
        try:
            selected_thumbnail = select_best_thumbnail(youtube_info, thumbnail_labels)
            thumbnail_path = os.path.join(thumbnail_folder, selected_thumbnail)
            
            if os.path.exists(thumbnail_path):
                upload_thumbnail(channel1, video_id, thumbnail_path)  # Using channel1 here
                print(f"Selected thumbnail based on content analysis: {selected_thumbnail}")
            else:
                print(f"Selected thumbnail {selected_thumbnail} not found in thumbnails folder")
        except Exception as e:
            print(f"Error selecting thumbnail with AI: {str(e)}")
            print("Falling back to random selection...")
            # Fallback to random selection if AI selection fails
            thumbnail_files = [f for f in os.listdir(thumbnail_folder) 
                             if f.lower().endswith(('.jpg', '.png'))]
            if thumbnail_files:
                random_thumbnail = os.path.join(thumbnail_folder, random.choice(thumbnail_files))
                upload_thumbnail(channel1, video_id, random_thumbnail)
                print(f"Selected thumbnail (random fallback): {random_thumbnail}")
            else:
                print("No thumbnail images found in thumbnails folder")

if __name__ == "__main__":
    main()