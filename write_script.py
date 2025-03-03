from together import Together
from dotenv import load_dotenv
import os
from datetime import datetime
import json
import google.generativeai as genai
from pydantic import BaseModel

load_dotenv(override=True)

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

science_topics = open("next_topics.txt", "r", encoding='utf-8').read().split("\n")

amazing_script = open("amazing_script.txt", "r", encoding='utf-8').read()

trending_context = open("context/grounding.txt", "r", encoding='utf-8').read()

your_memories = open("context/memory.txt", "r", encoding='utf-8').read()

class Topic(BaseModel):
    topic: str
    description: str

def clean_script_for_tts(script):
    # Remove the thinking section
    if "<think>" in script:
        script = script[script.find("</think>") + 8:]
    
    # Split into lines for processing
    lines = script.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip only specific formatting lines
        if line.strip() in [
            "---",
            "Script Notes:",
            "INTRO",
            "MAIN CONTENT",
            "CONCLUSION",
            "OUTRO"
        ]:
            continue
            
        # Remove asterisks but keep the content between them
        line = line.replace("*", "")
        
        # Skip empty lines
        if line.strip():
            cleaned_lines.append(line.strip())
    
    # Join the lines back together with proper spacing
    return "\n".join(cleaned_lines)

def generate_youtube_script():
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    prompt = f"""
     \n\n
    <Your latest knowledge of AI landscape>
    {your_memories}
    </Your latest knowledge of AI landscape>
    \n\n
    <Topics>
    {science_topics}
    </Topics>
    \n\n
    <Latest News>
    {trending_context}
    </Latest News>
    \n\n
    VERY IMPORTANT RULES:
    0. Pick ONE topic from the <Topics> list above. Prefer the topcis that alligns with the <Latest News>
    1. Write ONLY the spoken narrative
    2. Do not include any formatting, markers, or special characters
    3. Do not include stage directions or sound effects
    4. Avoid using colons, asterisks, or brackets
    5. Structure the content logically but don't include section headers etc
    6. The script is for information and NOVEL Thoughts on the topic. NOT a story told to people. So dont use "Folks we got something" etc.
    7. USE QUOTES FROM THE  <Latest News> IF AVAILABLE.
    8. Give your NOVEL personal opinions on topics to try to understand and express to the viewer some new insights (In first person (I think, I believe, My thoughts are, etc))
    9. AIM for 1000 - 2000 WORDS (IMPORTANT)
    10. DONT USE OVER THE TOP CRINGE LANGUAGE like: 
        - "this will change the world forever"
        - "this is the most important thing that will ever happen"
        - "This isnt just another model release—its a strategic earthquake shaking the foundations of the AI industry."
        - "bombshell announcement"
        - "earthquake shaking the foundations of the AI industry"
        - "seismic shift"
        - "strategic earthquake"
        - etc...

    11. End with a YouTube-style outro: "Stay tuned, give this video a like, and consider subscribing."

    Today date: {datetime.now().strftime("%Y-%m-%d")}
    \n\n
    The script should be conversational, NOVEL and engaging with clear sections.
    Include an attention-grabbing introduction (State the main topic from <Latest News> at the first sentence), main points, and a strong conclusion with the specified YouTube outro.

    Make it sound natural and engaging, focus on keeping peoples attention, building up tension and telling the latest news in an interesting way.
    MUST FOLLOW THE RULES (VERY IMPORTANT).

    Write the 1000 - 2000 WORDS script in a text with ONLY the words the voice actor will read, integrate quotes where it fits well and bring your NOVEL take on the <Latest News>, while still telling the latest news. Start with "Elon Musk just broke the news that Grok 3 is coming on February 17th.". 
    """


    messages = [
        {"role": "system", "content": f"""You are a script writer and AI / Futurist expert for the The Paradox Files YouTube Channel
         You concider your self a pro effective accerationist. This will be reflected in the scripts regarding the topics in either a positive or negative way.
         You must write the script in a way that include your NOVEL thoughts and opinions on the topic.
         \n\n
         <Your latest knowledge of AI landscape>
         {your_memories}
         </Your latest knowledge of AI landscape>
         \n\n
         <Example of a good on the point script>
         {amazing_script}
         </Example of a good on the point script>
         \n\n
         Give your hot takes and personal opinions on topics to ENGAGE the viewer. (In first person (I think, I believe, My thoughts are, etc)).
         The script is for information, not a story told to people. So dont use "Folks we got something" etc.
         ALWAYS have some of your NOVEL opinions and comments on the topic and  the <Latest News>.
         Use METAPHORICAL language to explain complex topics.
         We cant mention COVID-19, Coronavirus or Suicide in the script.
         AIM for 1000 - 2000 WORDS (IMPORTANT)
         DONT USE OVER THE TOP CRINGE LANGUAGE like: 
         - "this will change the world forever"
         - "this is the most important thing that will ever happen"
         - "This isnt just another model release—its a strategic earthquake shaking the foundations of the AI industry."
         """},
        {"role": "user", "content": prompt}


        ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            temperature=0.5,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            max_tokens=4000,
            stream=True
        )

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
        return cleaned_script

    except Exception as e:
        print(f"\nError during script generation: {str(e)}")
        return None

def extract_topic_from_script(script):
    model = genai.GenerativeModel('gemini-2.0-flash')
    prompt = f"""
    Script:
    {script}
    \n\n
    Extract the main topic and its description from this script above.
    Return only the topic title and description in JSON format with fields 'topic' and 'description'."""
    
    response = model.generate_content(prompt)
    
    response_text = response.text
    try:
        import json
        response_dict = json.loads(response_text)
        return Topic(**response_dict)
    except:
        # Fallback if JSON parsing fails
        return Topic(topic="Unknown Topic", description="Unable to extract topic")

def main():
    # Load environment variables
    load_dotenv(override=True)
        
    # Generate the script
    script = generate_youtube_script()
        
    # Save the generated script
    with open("generated_script.txt", "w", encoding='utf-8') as file:
        file.write(script)
            
    print("\n\nScript has been generated and saved to 'generated_script.txt'")
        
    # Extract topic using structured output
    extracted_topic = extract_topic_from_script(script)
    print(f"\n\nTopic extracted from script: {extracted_topic.topic}")
    
    # Update science_topics_covered.json
    try:
        with open("science_topics_covered.json", "r") as f:
            topics_data = json.load(f)
            
        # Add new topic if not already present
        if extracted_topic.topic not in topics_data["topics_already_covered"]:
            topics_data["topics_already_covered"].append(extracted_topic.topic)
            
        # Save updated topics
        with open("science_topics_covered.json", "w") as f:
            json.dump(topics_data, f, indent=4, ensure_ascii=False)
            
        print(f"\nUpdated science_topics_covered.json with new topic")
    except Exception as e:
        print(f"\nError updating topics file: {str(e)}")

if __name__ == "__main__":
    main()
