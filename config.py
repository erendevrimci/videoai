"""
VideoAI Configuration Module

This module defines the configuration settings for the VideoAI project using Pydantic models
to ensure type safety and validation. It loads settings from environment variables and
provides default values for all settings.
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

from file_manager import FileManager

# Load environment variables
load_dotenv(override=True)

# Initialize the file manager
file_mgr = FileManager()

# Base project directory (use file manager's base_dir)
BASE_DIR = file_mgr.base_dir

class OpenAISettings(BaseModel):
    """OpenAI API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    script_model: str = Field(default="gpt-4o")
    title_desc_model: str = Field(default="gpt-4o")
    video_edit_model: str = Field(default="o3-mini")
    temperature: float = Field(default=0.7)
    max_tokens: int = Field(default=4000)

class ElevenLabsSettings(BaseModel):
    """ElevenLabs API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("ELEVENLABS_API_KEY", ""))
    model_id: str = Field(default="eleven_turbo_v2")
    default_voice_id: str = Field(default="29vD33N1CtxCmqQRPOHJ")
    stability: float = Field(default=0.35)
    similarity_boost: float = Field(default=0.55)
    style: float = Field(default=0.10)
    use_speaker_boost: bool = Field(default=True)

class YoutubeSettings(BaseModel):
    """YouTube API settings"""
    api_key: str = Field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY", ""))
    client_id: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_ID", ""))
    client_secret: str = Field(default_factory=lambda: os.getenv("YOUTUBE_CLIENT_SECRET", ""))
    privacy_status: str = Field(default="private")
    category_id: str = Field(default="28")  # Science & Technology

class ChannelConfig(BaseModel):
    """Channel-specific configuration"""
    name: str
    voice_id: str
    youtube_credentials_file: str
    youtube_info_file: str
    privacy_status: str = Field(default="private")

class FilePathConfig(BaseModel):
    """File path configuration"""
    script_file: str = Field(default="generated_script.txt")
    voice_file: str = Field(default="voice/generated_voice.mp3")
    captions_file: str = Field(default="generated_voice.srt")
    output_video_file: str = Field(default="output_video.mp4")
    final_video_file: str = Field(default="final_output.mp4")
    final_subtitled_video_file: str = Field(default="final_output_with_subtitles.mp4")
    clips_metadata_file: str = Field(default="clips/clips_label.md")
    clips_directory: str = Field(default="clips")
    background_music_directory: str = Field(default="background_music")
    output_directory: str = Field(default="outputs")

class VideoEditConfig(BaseModel):
    """Video editing configuration"""
    max_clip_duration: int = Field(default=10)
    default_clip_duration: int = Field(default=10)
    voice_volume: float = Field(default=1.4)
    background_music_volume: float = Field(default=0.1)
    subtitle_font: str = Field(default="Arial")
    subtitle_font_size: int = Field(default=18)

class ScriptGenerationConfig(BaseModel):
    """Script generation configuration"""
    min_words: int = Field(default=100)
    max_words: int = Field(default=200)
    target_audience: str = Field(default="tech enthusiasts and AI learners")
    tone: str = Field(default="informative yet conversational")
    style: str = Field(default="clear, engaging, and accessible")

class AppConfig(BaseSettings):
    """Main application configuration"""
    # API settings
    openai: OpenAISettings = Field(default_factory=OpenAISettings)
    elevenlabs: ElevenLabsSettings = Field(default_factory=ElevenLabsSettings)
    youtube: YoutubeSettings = Field(default_factory=YoutubeSettings)
    
    # File paths
    file_paths: FilePathConfig = Field(default_factory=FilePathConfig)
    
    # Video editing settings
    video_edit: VideoEditConfig = Field(default_factory=VideoEditConfig)
    
    # Script generation settings
    script_generation: ScriptGenerationConfig = Field(default_factory=ScriptGenerationConfig)
    
    # Channel configurations
    channels: Dict[int, ChannelConfig] = Field(default_factory=lambda: {
        1: ChannelConfig(
            name="Channel 1",
            voice_id="29vD33N1CtxCmqQRPOHJ",
            youtube_credentials_file="youtube_token_channel1.json",
            youtube_info_file="youtube_info_channel1.json"
        ),
        2: ChannelConfig(
            name="Channel 2",
            voice_id="EXAVITQu4vr4xnSDxMaL",
            youtube_credentials_file="youtube_token_channel2.json",
            youtube_info_file="youtube_info_channel2.json"
        ),
        3: ChannelConfig(
            name="Channel 3",
            voice_id="BLaQKPB2UVQ1JfmZQYQn",
            youtube_credentials_file="youtube_token_channel3.json",
            youtube_info_file="youtube_info_channel3.json"
        )
    })
    
    # Default channel to use if not specified
    default_channel: int = Field(default=1)
    
    model_config = {
        "env_file": ".env",
        "env_nested_delimiter": "__",
        "extra": "ignore"  # Ignore extra fields from environment variables
    }

# Create a global config instance
config = AppConfig()

# Helper function to get channel-specific config
def get_channel_config(channel_number: Optional[int] = None) -> ChannelConfig:
    """Get configuration for a specific channel"""
    channel = channel_number or config.default_channel
    if channel in config.channels:
        return config.channels[channel]
    raise ValueError(f"Invalid channel number: {channel}")

# Legacy helper functions that now use FileManager internally
def get_abs_path(rel_path: str) -> Path:
    """
    Convert a relative path to absolute path using FileManager.
    
    This is maintained for backward compatibility with existing code.
    New code should use FileManager.get_abs_path() directly.
    """
    return file_mgr.get_abs_path(rel_path)
    
def get_channel_output_path(channel_number: Optional[int] = None) -> Path:
    """
    Get the output directory path for a specific channel using FileManager.
    
    This is maintained for backward compatibility with existing code.
    New code should use FileManager.get_channel_output_path() directly.
    """
    return file_mgr.get_channel_output_path(channel_number or config.default_channel)

if __name__ == "__main__":
    # Print configuration for debugging
    print(f"Configuration loaded from {__file__}")
    print(f"Base directory: {BASE_DIR}")
    
    # Example of accessing configuration values
    print("\nOpenAI Configuration:")
    print(f"  Model: {config.openai.script_model}")
    
    print("\nElevenLabs Configuration:")
    print(f"  Default Voice ID: {config.elevenlabs.default_voice_id}")
    
    print("\nFile Paths:")
    print(f"  Script: {get_abs_path(config.file_paths.script_file)}")
    print(f"  Voice: {get_abs_path(config.file_paths.voice_file)}")
    
    print("\nChannel Configurations:")
    for channel_num, channel_config in config.channels.items():
        print(f"  Channel {channel_num}: {channel_config.name}")
        print(f"    Voice ID: {channel_config.voice_id}")
        print(f"    YouTube Credentials: {channel_config.youtube_credentials_file}")
        
    # Example of using FileManager directly
    print("\nFileManager Examples:")
    print(f"  Channel 1 Script Path: {file_mgr.get_script_path(1, 'script')}")
    print(f"  Channel 2 Voice Path: {file_mgr.get_audio_output_path(2, 'generated_voice')}")
    print(f"  Channel 3 Video Path: {file_mgr.get_video_output_path(3, 'final_output')}")