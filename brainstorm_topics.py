from together import Together
from dotenv import load_dotenv
import os
from datetime import datetime
from dotenv import load_dotenv
import json

load_dotenv(override=True)

category = "ai"

topics_already_covered = json.load(open(f"categories/{category}/topics_covered.json"))

trending_context = open("context/grounding.txt", "r", encoding='utf-8').read()
sub_category = open(f"categories/{category}/sub_category.txt", "r", encoding='utf-8').read()

def generate_ideas():

    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))
    
    prompt = f"""
    Exact Ideas I Already Covered:
    \n\n
    {topics_already_covered}
    \n\n
    Your task is to brainstrom super interesting ideas about what can happen in near future in the topics below:
    {sub_category}

     that are not already covered in the list above.
     \n\n
     <Trending Context>
     {trending_context}
     </Trending Context>
     \n\n
     Also take the trending context above into account:

    Dont suggest exact ideas that are already covered.

    But do suggest simple ideas that are super interesting and engaging in the style of the topics above.
    \n\n
    Make sure the ideas are simple to understand, but cause FEAR / SCARINESS / NEGATIVE EMOTIONS in the viewer.
    \n\n
    ALWAYS Try to integrate A MAJOR LAB / COMPANY / PERSON INTO THE IDEA. (OPENAI, SAM ALTMAN, Elon Musk, TESLA, DEEPSEEK, GOOGLE, NVIDIA, ETC.)
    \n\n
    Make sure the ideas are not already covered.
    \n\n
    Prefer ideas on the trending context if they align with our preferences.
    \n\n
    All ideas should start with "What could happen if", "What if", "Could", "Can" etc.
    \n\n
    Write 3 new realistic ideas about the <Trending Context> above that can happen in near future, and that we can use for the next video:
    """


    messages = [
        {"role": "system", "content": f"""
        You are a brainstorming expert. 
        You are tasked with brainstorming super interesting {category} ideas that are not already covered.
        You are also tasked with making sure the ideas are super interesting and engaging idea for a youtube video.
        The realistic ideas should be focused on what could happen in a worse case scenario.

        """},

        {"role": "user", "content": prompt}


        ]
    
    try:
        response = client.chat.completions.create(
            model="deepseek-ai/DeepSeek-R1",
            messages=messages,
            temperature=0.7,
            top_p=0.7,
            top_k=50,
            repetition_penalty=1,
            max_tokens=2000,
            stream=True

        )

        script_content = ""
        print("\nGenerating ideas...\n")
        
        for chunk in response:
            if hasattr(chunk, 'choices') and len(chunk.choices) > 0:
                if hasattr(chunk.choices[0], 'delta') and hasattr(chunk.choices[0].delta, 'content'):
                    current_content = chunk.choices[0].delta.content
                    if current_content:
                        script_content += current_content
                        print(current_content, end="", flush=True)
        
        # Filter out the thinking process and clean up the response
        filtered_content = script_content
        if '<think>' in filtered_content:
            filtered_content = filtered_content.split('</think>')[1]  # Get content after </think>
        
        # Clean up the text by removing duplicates and empty lines
        lines = filtered_content.strip().split('\n')
        cleaned_lines = []
        seen = set()
        
        for line in lines:
            line = line.strip()
            if line and line not in seen:
                cleaned_lines.append(line)
                seen.add(line)
        
        # Join the cleaned lines back together
        final_content = '\n'.join(cleaned_lines)
        
        # Save to text file
        try:
            with open(f'categories/{category}/next_topics.txt', 'w', encoding='utf-8') as f:
                f.write(final_content)
            print("\nTopics saved to next_topics.txt")
        except Exception as e:
            print(f"\nError saving text file: {str(e)}")
        
    except Exception as e:
        print(f"\nError during script generation: {str(e)}")
        return None
    
def main():
    generate_ideas()

if __name__ == "__main__":
    main()

