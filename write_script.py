"""
Script generation module for VideoAI.

This module handles the generation of video scripts using AI models,
extracting topics, and maintaining the topics database.
"""

import os
from datetime import datetime
import json
from typing import Optional, List, Dict, Any
from pathlib import Path

from together import Together
import google.generativeai as genai
from pydantic import BaseModel

from config import config, get_channel_config
from file_manager import FileManager

# Initialize the file manager
file_mgr = FileManager()

# Configure APIs
def configure_apis():
    """Configure API settings for script generation."""
    # Configure Gemini
    gemini_api_key = os.getenv("GEMINI_API_KEY") or ""
    genai.configure(api_key=gemini_api_key)


# Pydantic model for topic data
class Topic(BaseModel):
    """Topic model for consistent data handling."""
    topic: str
    description: str


def load_input_files(category: str = "ai") -> Dict[str, str]:
    """
    Load all required input files for script generation.
    
    Args:
        category (str): Category to use for topics
        
    Returns:
        Dict[str, str]: Dictionary of content from each file
    """
    # Construct paths using FileManager
    category_path = file_mgr.get_abs_path(f"categories/{category}")
    topics_file = category_path / "next_topics.txt"
    script_example_file = file_mgr.get_abs_path("amazing_script.txt")
    context_file = file_mgr.get_abs_path("context/grounding.txt")
    memory_file = file_mgr.get_abs_path("context/memory.txt")
    
    # Load file contents with error handling using FileManager
    files = {}
    
    # Read topics file
    topics_content = file_mgr.read_text(topics_file)
    if topics_content is None:
        print(f"Error: Could not read topics file: {topics_file}")
        raise FileNotFoundError(f"Could not read topics file: {topics_file}")
    files["topics"] = topics_content.split("\n")
    
    # Read script example file
    amazing_script = file_mgr.read_text(script_example_file)
    if amazing_script is None:
        print(f"Error: Could not read script example file: {script_example_file}")
        raise FileNotFoundError(f"Could not read script example file: {script_example_file}")
    files["amazing_script"] = amazing_script
    
    # Read context file
    trending_context = file_mgr.read_text(context_file)
    if trending_context is None:
        print(f"Error: Could not read context file: {context_file}")
        raise FileNotFoundError(f"Could not read context file: {context_file}")
    files["trending_context"] = trending_context
    
    # Read memory file
    your_memories = file_mgr.read_text(memory_file)
    if your_memories is None:
        print(f"Error: Could not read memory file: {memory_file}")
        raise FileNotFoundError(f"Could not read memory file: {memory_file}")
    files["your_memories"] = your_memories
    
    return files

def clean_script_for_tts(script: str) -> str:
    """
    Clean the generated script for text-to-speech processing.
    
    Args:
        script (str): The raw script text
        
    Returns:
        str: Cleaned script ready for TTS
    """
    # Remove the thinking section if present
    if "<think>" in script:
        script = script[script.find("</think>") + 8:]
    
    # Split into lines for processing
    lines = script.split('\n')
    cleaned_lines = []
    
    # Identify formatting lines to skip
    formatting_markers = [
        "---",
        "Script Notes:",
        "INTRO",
        "MAIN CONTENT",
        "CONCLUSION",
        "OUTRO"
    ]
    
    for line in lines:
        # Skip formatting lines
        if line.strip() in formatting_markers:
            continue
            
        # Remove asterisks but keep the content between them
        line = line.replace("*", "")
        
        # Skip empty lines
        if line.strip():
            cleaned_lines.append(line.strip())
    
    # Join the lines back together with proper spacing
    return "\n".join(cleaned_lines)

def generate_youtube_script(input_files: Dict[str, Any] = None) -> Optional[str]:
    """
    Generate a YouTube script using the Together AI API.
    
    Args:
        input_files (Dict[str, Any], optional): Dictionary with input file contents
        
    Returns:
        Optional[str]: The generated script, or None if generation failed
    """
    # Load input files if not provided
    if input_files is None:
        input_files = load_input_files(category="ai")
    
    # Get API keys from environment or config
    together_api_key = os.getenv("TOGETHER_API_KEY") or config.openai.api_key
    
    # Initialize Together client
    client = Together(api_key=together_api_key)
    
    # Get data from input files
    your_memories = input_files["your_memories"]
    topics = input_files["topics"]
    trending_context = input_files["trending_context"]
    amazing_script = input_files["amazing_script"]
    
    # Format topics for prompt
    topics_text = "\n".join(topics) if isinstance(topics, list) else topics
    
    # Get script generation settings from config
    min_words = config.script_generation.min_words
    max_words = config.script_generation.max_words
    target_audience = config.script_generation.target_audience
    tone = config.script_generation.tone
    style = config.script_generation.style
    
    # Construct the prompt
    prompt = f"""
     \n\n
    <Your latest knowledge>
    {your_memories}
    </Your latest knowledge>
    \n\n
    <Topics>
    {topics_text}
    </Topics>
    \n\n
    <Latest News>
    {trending_context}
    </Latest News>
    \n\n
    VERY IMPORTANT RULES:
    0. Pick ONE topic from the <Topics> list above. Prefer the topics that align with the <Latest News>
    1. Write ONLY the spoken narrative
    2. Do not include any formatting, markers, or special characters
    3. Do not include stage directions or sound effects
    4. Avoid using colons, asterisks, or brackets
    5. Structure the content logically but don't include section headers etc
    6. The script is for information and NOVEL Thoughts on the topic. NOT a story told to people. So don't use "Folks we got something" etc.
    7. USE QUOTES FROM THE <Latest News> IF AVAILABLE.
    8. Give your NOVEL personal opinions on topics to try to understand and express to the viewer some new insights (In first person (I think, I believe, My thoughts are, etc))
    9. AIM for {min_words} - {max_words} WORDS (IMPORTANT)
    10. DONT USE OVER THE TOP CRINGE LANGUAGE like: 
        - "this will change the world forever"
        - "this is the most important thing that will ever happen"
        - "This isn't just another model release—its a strategic earthquake shaking the foundations of the AI industry."
        - "bombshell announcement"
        - "earthquake shaking the foundations of the AI industry"
        - "seismic shift"
        - "strategic earthquake"
        - etc...

    11. End with a YouTube-style outro: "Stay tuned, give this video a like, and consider subscribing."

    Today date: {datetime.now().strftime("%Y-%m-%d")}
    \n\n
    The script should be {tone}, NOVEL and engaging with clear sections.
    Include an attention-grabbing introduction (State the main topic from <Latest News> at the first sentence), main points, and a strong conclusion with the specified YouTube outro.

    Make it sound natural and engaging, focus on keeping people's attention, building up tension and telling the latest news in an interesting way.
    MUST FOLLOW THE RULES (VERY IMPORTANT).

    Write the {min_words} - {max_words} WORDS script in a text with ONLY the words the voice actor will read, integrate quotes where it fits well and bring your NOVEL take on the <Latest News>, while still telling the latest news.". 
    """

    # Construct messages for the API
    messages = [
        {"role": "system", "content": f"""You are a script writer and an expert for the regarding YouTube Channel
         You consider yourself a pro effective accelerationist. This will be reflected in the scripts regarding the topics in either a positive or negative way.
         You must write the script in a way that includes your NOVEL thoughts and opinions on the topic.
         \n\n
         <Your latest knowledge>
         {your_memories}
         </Your latest knowledge>
         \n\n
         <Example of a good on the point script>
         {amazing_script}
         </Example of a good on the point script>
         \n\n
         Give your hot takes and personal opinions on topics to ENGAGE the viewer. (In first person (I think, I believe, My thoughts are, etc)).
         The script is for information, not a story told to people. So don't use "Folks we got something" etc.
         ALWAYS have some of your NOVEL opinions and comments on the topic and the <Latest News>.
         Use METAPHORICAL language to explain complex topics.
         We can't mention COVID-19, Coronavirus or Suicide in the script.
         AIM for {min_words} - {max_words} WORDS (IMPORTANT)
         DONT USE OVER THE TOP CRINGE LANGUAGE like: 
         - "this will change the world forever"
         - "this is the most important thing that will ever happen"
         - "This isn't just another model release—its a strategic earthquake shaking the foundations of the AI industry."
         """},
        {"role": "user", "content": prompt}
    ]
    
    try:
        # Use model from config
        model = "deepseek-ai/DeepSeek-R1"
        temperature = config.openai.temperature
        max_tokens = config.openai.max_tokens
        
        print(f"\nGenerating script using {model}...\n")
        print(f"Target word count: {min_words}-{max_words} words\n")
        
        # Make API request with streaming
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            max_tokens=max_tokens,
            stream=True
        )

        # Process streaming response
        script_content = ""
        print("\nGenerating script...\n")
        
        for chunk in response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    current_content = chunk.choices[0].delta.content
                    if current_content:
                        script_content += current_content
                        print(current_content, end="", flush=True)
        
        # Clean the script for TTS
        cleaned_script = clean_script_for_tts(script_content)
        
        # Print word count info
        word_count = len(cleaned_script.split())
        print(f"\n\nScript generated - {word_count} words")
        
        return cleaned_script

    except Exception as e:
        print(f"\nError during script generation: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def extract_topic_from_script(script: str) -> Topic:
    """
    Extract the main topic and description from a script using Gemini AI.
    
    Args:
        script (str): The script content
        
    Returns:
        Topic: A Topic model with the extracted information
    """
    # Configure Gemini if not already done
    configure_apis()
    
    # Select model and configure prompt
    model_name = 'gemini-2.0-flash'
    model = genai.GenerativeModel(model_name)
    
    prompt = f"""
    Script:
    {script}
    \n\n
    Extract the main topic and its description from this script above.
    Return only the topic title and description in JSON format with fields 'topic' and 'description'.
    Example format:
    {{"topic": "Topic Title", "description": "Brief description of the topic"}}
    """
    
    try:
        print(f"\nExtracting topic using {model_name}...")
        response = model.generate_content(prompt)
        
        # Process the response
        response_text = response.text.strip()
        
        # Clean the JSON response
        if response_text.startswith('```json'):
            response_text = response_text[7:-3]
        elif response_text.startswith('```'):
            response_text = response_text[3:-3]
            
        response_text = response_text.strip()
        response_dict = json.loads(response_text)
        
        # Check for required fields
        if not all(key in response_dict for key in ['topic', 'description']):
            raise ValueError("Missing required fields in JSON response")
        
        # Create Topic object
        topic = Topic(**response_dict)
        print(f"Extracted topic: {topic.topic}")
        
        return topic
    
    except Exception as e:
        print(f"Topic extraction error: {str(e)}")
        # Return a default topic in case of errors
        return Topic(topic="Unknown Topic", description="Unable to extract topic")

def update_topics_covered(topic: Topic, topics_file: str = "topics_covered.json") -> bool:
    """
    Update the topics_covered.json file with a new topic.
    
    Args:
        topic (Topic): The topic to add
        topics_file (str): Path to the topics file
        
    Returns:
        bool: True if update was successful, False otherwise
    """
    topics_path = file_mgr.get_abs_path(topics_file)
    
    try:
        # Load existing topics using FileManager
        topics_data = file_mgr.read_json(topics_path)
        if topics_data is None:
            print(f"Error: Could not read topics file: {topics_path}")
            return False
        
        # Add new topic if not already present
        if topic.topic not in topics_data["topics_already_covered"]:
            topics_data["topics_already_covered"].append(topic.topic)
            
            # Save updated topics using FileManager
            success = file_mgr.write_json(topics_path, topics_data)
            if success:
                print(f"Updated {topics_file} with new topic: {topic.topic}")
                return True
            else:
                print(f"Error: Failed to write topics file: {topics_path}")
                return False
        else:
            print(f"Topic '{topic.topic}' already exists in {topics_file}")
            return True
            
    except Exception as e:
        print(f"Error updating topics file: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def save_script(script: str, channel_number: Optional[int] = None) -> Optional[Path]:
    """
    Save the generated script to a file.
    
    Args:
        script (str): The script content to save
        channel_number (Optional[int]): Channel number to use, or None for default
        
    Returns:
        Optional[Path]: Path to the saved script file, or None if saving failed
    """
    if script is None:
        print("No script to save.")
        return None
        
    try:
        # Get script file path from config
        script_path = file_mgr.get_abs_path(config.file_paths.script_file)
        
        # First save to channel-specific directory if channel is specified
        if channel_number is not None:
            # Use FileManager's get_script_path to get the correct channel script path
            channel_script_path = file_mgr.get_script_path(channel_number, "script")
            
            # Save to channel-specific directory using FileManager
            success = file_mgr.write_text(channel_script_path, script)
            if success:
                print(f"Script saved to channel directory: {channel_script_path}")
                # Also save a copy with a different name if needed
                alt_script_path = file_mgr.get_script_path(channel_number, "generated_script")
                file_mgr.write_text(alt_script_path, script)
            else:
                print(f"Error: Failed to save script to channel directory: {channel_script_path}")
        
        # Save to main script file using FileManager
        success = file_mgr.write_text(script_path, script)
        if success:
            print(f"Script saved to: {script_path}")
            return script_path
        else:
            print(f"Error: Failed to save script to main location: {script_path}")
            # If we saved to channel path earlier, return that instead
            if channel_number is not None:
                return file_mgr.get_script_path(channel_number, "script")
            return None
        
    except Exception as e:
        print(f"Error saving script: {str(e)}")
        return None


def main(channel_number: Optional[int] = None) -> None:
    """
    Main function to generate a script, extract the topic, and update topics database.
    
    Args:
        channel_number (Optional[int]): Channel number to use, or None for default
    """
    # Configure APIs
    configure_apis()
    
    print("Starting script generation process...")
    
    try:
        # Load input files
        input_files = load_input_files(category="ai")
        
        # Generate the script
        script = generate_youtube_script(input_files)
        
        if script is None:
            print("Script generation failed.")
            return
            
        
            
        # Extract topic from script
        extracted_topic = extract_topic_from_script(script)
        
        # Update topics database
        update_topics_covered(extracted_topic)


        # Save the generated script
        script_path = save_script(script, channel_number)
        
        if script_path is None:
            print("Failed to save script.")
            return
        
        print("\nScript generation completed successfully.")
        
    except Exception as e:
        print(f"Error in script generation process: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Parse command line arguments
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate a script for VideoAI")
    parser.add_argument("--channel", type=int, choices=[1, 2, 3], 
                       help="Channel number to use (1-3)")
    
    args = parser.parse_args()
    
    main(args.channel)
