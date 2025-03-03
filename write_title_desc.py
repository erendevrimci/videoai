import os
import json
from dotenv import load_dotenv
import google.generativeai as genai
from pydantic import BaseModel

load_dotenv(override=True)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Define the schema for YouTube info
class YouTubeInfo(BaseModel):
    title: str
    description: str

def generate_title(script: str) -> str:
    title_prompt = f"""You are a YouTube title expert. Create an extremely engaging, clickbait-style title for this video that will maximize clicks.T
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
- CLAUDE 4 Just SHOCKED OpenAI’s GPT-5 Plans!
- AI AGENTS Will Disrupt ALL KNOWLEDGE Work! Are You at RISK?
ALWAYS MENTION A COMPANY OR WELL KNOWN ENTITY IN THE TITLE IF POSSIBLE.

Generate only the extremely clickbait-style title in the style of the examples above, nothing else. NEVER USE EMOJIS. Keep it under 75 characters."""


    model = genai.GenerativeModel('gemini-2.0-flash')
    title_response = model.generate_content(
        title_prompt.format(script=script),
        generation_config=genai.GenerationConfig(temperature=1)
    )
    title = title_response.text.strip()
    return title

def generate_desc(script: str) -> str:
    desc_prompt = f"""
Write a YouTube description in exactly this format:
[Your engaging description text here] Like & Subscribe for more!

Rules:
- Must be under 300 characters total
- Must end with "Like & Subscribe for more!"
- Just the description text
- NO Emojis
- Mention the Youtube Channel "All About AI" in the description. Url: https://www.youtube.com/@AllAboutAI

Context:
{script}

Write a SEO YouTube description for this video:
"""

    model = genai.GenerativeModel('gemini-2.0-flash')
    desc_response = model.generate_content(
        desc_prompt,
        generation_config=genai.GenerationConfig(temperature=0.7)
    )
    description = desc_response.text.strip()
    return description

def save_youtube_info(title: str, description: str) -> None:
    """Save the generated title and description to a JSON file."""
    youtube_info = YouTubeInfo(title=title, description=description)
    
    with open("youtube_info.json", "w", encoding="utf-8") as f:
        json.dump(youtube_info.model_dump(), f, indent=2, ensure_ascii=False)

def main():
    # Read the generated script
    with open("generated_script.txt", "r", encoding="utf-8") as f:
        script = f.read()
    
    # Generate title and description
    title = generate_title(script)
    desc = generate_desc(script)
    
    # Save to JSON file
    save_youtube_info(title, desc)
    
    # Print results
    print("\nGenerated Title:")
    print(title)
    print("\nGenerated Description:")
    print(desc)
    print("\nSaved to youtube_info.json")

if __name__ == "__main__":
    main()
