import os
from openai import OpenAI
from dotenv import load_dotenv


load_dotenv(override=True)

# Ensure the OpenAI API key is configured
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def generate_subtitles(audio_file_path: str, output_srt_path: str) -> None:

    """
    Generate subtitles (SRT file) from the given audio file using the OpenAI Whisper API.
    
    This function sends the generated voice mp3 file to the Whisper transcription API
    and requests the transcription in SRT format. The resulting subtitles are saved 
    into the output file.
    
    Args:
        audio_file_path (str): The path to the audio file (e.g., a generated_voice.mp3).
        output_srt_path (str): Path to save the SRT file.
    """
    try:
        with open(audio_file_path, "rb") as audio_file:
            print("Transcribing audio using OpenAI Whisper API...")
            # Request transcription with SRT output format
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="srt"
            )

            
        # transcription is a string containing the SRT formatted text.
        with open(output_srt_path, "w", encoding="utf-8") as srt_file:
            srt_file.write(transcription)
            
        print(f"Subtitles successfully saved to {output_srt_path}")
    except Exception as e:
        print("An error occurred during transcription:", e)

if __name__ == "__main__":
    # Path to the generated voice file (adjust if needed)
    audio_file_path = "voice/generated_voice.mp3"
    # Output SRT file path
    output_srt_path = "generated_voice.srt"
    generate_subtitles(audio_file_path, output_srt_path)
