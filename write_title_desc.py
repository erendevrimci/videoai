"""
Title and Description Generation Module

This module generates SEO-optimized titles and descriptions for YouTube videos
using the Gemini AI model.
"""

import os
import json
import traceback
from pathlib import Path
import google.generativeai as genai
from pydantic import BaseModel, Field
from config import config, get_channel_config
from file_manager import FileManager

# Initialize the file manager
file_mgr = FileManager()

# Define the schema for YouTube info
class YouTubeInfo(BaseModel):
    """YouTube metadata schema for title and description"""
    title: str = Field(..., description="Video title (clickbait style)")
    description: str = Field(..., description="Video description with SEO optimization")
    tags: list[str] = Field(default=["ai", "openai", "llms", "robotics", "future", "tech"], 
                           description="Tags for the video")

def initialize_gemini_client():
    """Initialize the Gemini AI client with API key from config"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API key not found in environment variables")
    
    genai.configure(api_key=gemini_api_key)

def generate_title(script: str) -> str:
    """
    Generate an engaging, clickbait-style title for a YouTube video.
    
    Args:
        script: The video script text
        
    Returns:
        A clickbait-style title for YouTube
    """
    title_prompt = f"""You are a YouTube title expert. Create an extremely engaging, clickbait-style title for this video that will maximize clicks.
    The title should be emotional, create fear, and highlight the most dramatic aspects.
    ALWAYS MENTION A COMPANY OR WELL KNOWN ENTITY IN THE TITLE IF POSSIBLE.

Context:
{script}

Examples of good titles:
- Sam Altman CONFIRMS GPT-5 and SCRAPS o3-model
- Sam Altman: 2025 Will Be The End of Software Engineering As We Know It
- The TRUTH About Elon Musk's $97B Bid On Sam Altman`s OpenAI
- Sam Altman's "Three Observations": The Truth About AI Agents and Job Loss?
- OpenAI Abandons O3 For GPT-5 In MASSIVE Strategy Shift!
- AI AGENTS Will Challenge Humans For Their Job
- OpenAI's NIGHTMARE: Claude 4 is COMING!
- CLAUDE 4 Just SHOCKED OpenAI's GPT-5 Plans!
- AI AGENTS Will Disrupt ALL KNOWLEDGE Work! Are You at RISK?
ALWAYS MENTION A COMPANY OR WELL KNOWN ENTITY IN THE TITLE IF POSSIBLE.

Generate only the extremely clickbait-style title in the style of the examples above, nothing else. NEVER USE EMOJIS. Keep it under 75 characters."""

    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        title_response = model.generate_content(
            title_prompt,
            generation_config=genai.GenerationConfig(temperature=1.0)
        )
        title = title_response.text.strip()
        return title
    except Exception as e:
        print(f"Error generating title: {str(e)}")
        print(traceback.format_exc())
        return "ERROR: Failed to generate title"

def generate_desc(script: str, channel_name: str = "All About AI", channel_url: str = "https://www.youtube.com/@AllAboutAI") -> str:
    """
    Generate an SEO-optimized description for a YouTube video.
    
    Args:
        script: The video script text
        channel_name: Name of the YouTube channel
        channel_url: URL of the YouTube channel
        
    Returns:
        A formatted YouTube description
    """
    desc_prompt = f"""
Write a YouTube description in exactly this format:
[Your engaging description text here] Like & Subscribe for more!

Rules:
- Must be under 300 characters total
- Must end with "Like & Subscribe for more!"
- Just the description text
- NO Emojis
- Mention the Youtube Channel "{channel_name}" in the description. Url: {channel_url}

Context:
{script}

Write a SEO YouTube description for this video:
"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    try:
        desc_response = model.generate_content(
            desc_prompt,
            generation_config=genai.GenerationConfig(temperature=0.7)
        )
        description = desc_response.text.strip()
        return description
    except Exception as e:
        print(f"Error generating description: {str(e)}")
        print(traceback.format_exc())
        return "ERROR: Failed to generate description. Like & Subscribe for more!"

def save_youtube_info(title: str, description: str, output_file: Path) -> bool:
    """
    Save the generated title and description to a JSON file.
    
    Args:
        title: The generated video title
        description: The generated video description
        output_file: Path to save the YouTube info JSON
        
    Returns:
        True if successful, False otherwise
    """
    try:
        # Create YouTubeInfo object with validation
        youtube_info = YouTubeInfo(title=title, description=description)
        
        # Convert to dictionary
        youtube_info_dict = youtube_info.model_dump()
        
        # Use FileManager to save JSON
        success = file_mgr.write_json(output_file, youtube_info_dict)
        return success
    except Exception as e:
        print(f"Error saving YouTube info: {str(e)}")
        print(traceback.format_exc())
        return False

def main(channel_number: int = None):
    """
    Main function to generate and save YouTube title and description.
    
    Args:
        channel_number: Optional channel number to use for configuration
    """
    try:
        # Use default channel if none specified
        if channel_number is None:
            channel_number = config.default_channel
            
        # Initialize Gemini client
        initialize_gemini_client()
        
        # Get channel-specific configuration
        channel_config = get_channel_config(channel_number)
        
        # Set file paths using FileManager
        script_file_paths = [
            file_mgr.get_script_path(channel_number, "script"),
            file_mgr.get_script_path(channel_number, "generated_script"),
            file_mgr.get_abs_path(config.file_paths.script_file)
        ]
        output_file = file_mgr.get_channel_output_path(channel_number) / channel_config.youtube_info_file
        
        print(f"Generating title and description for channel {channel_number}")
        
        # Try multiple script file paths
        script = None
        used_path = None
        
        for path in script_file_paths:
            script = file_mgr.read_text(path)
            if script is not None:
                used_path = path
                print(f"Reading script from: {used_path}")
                break
                
        if script is None:
            # If all paths failed, try generating a script first
            print(f"Error: Could not read script file from any path. Attempting to generate script...")
            
            import write_script
            write_script.main(channel_number)
            
            # Try again after script generation
            for path in script_file_paths:
                script = file_mgr.read_text(path)
                if script is not None:
                    used_path = path
                    print(f"Reading script from: {used_path}")
                    break
            
            if script is None:
                print("Error: Could not generate or read script file")
                return
        
        # Generate title and description
        print("Generating title...")
        title = generate_title(script)
        
        print("Generating description...")
        desc = generate_desc(script)
        
        # Save to JSON file
        if save_youtube_info(title, desc, output_file):
            print(f"YouTube info saved to {output_file}")
            
            # Also save to project root for compatibility
            root_output = file_mgr.get_abs_path("youtube_info.json")
            if file_mgr.write_json(root_output, {"title": title, "description": desc}):
                print(f"YouTube info also saved to {root_output}")
        else:
            print(f"Failed to save YouTube info to {output_file}")
        
        # Print results
        print("\nGenerated Title:")
        print(title)
        print("\nGenerated Description:")
        print(desc)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate YouTube title and description")
    parser.add_argument("--channel", type=int, help="Channel number to use for configuration")
    
    args = parser.parse_args()
    
    main(channel_number=args.channel)