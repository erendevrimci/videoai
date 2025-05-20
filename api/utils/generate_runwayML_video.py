from runwayml import RunwayML
import time
import os

def generate_runwayML_video(prompt: str, negativePrompt: str, duration: int, cfgScale: int, aspectRatio: str, startImage: str, endImage: str):
    api_key = os.getenv("RUNWAYML_API_KEY")
    client = RunwayML(api_key)
    task = client.text_to_video(
        prompt_text=prompt,
        duration=duration,
        ratio=aspectRatio,
        prompt_image=f"data:image/png;base64,{startImage}",
    )
    task_id = task.id
    time.sleep(10)

    task = client.tasks.retrieve(task_id)

    while task.status not in ['SUCCEEDED', 'FAILED']:
        time.sleep(10)
        task = client.tasks.retrieve(task_id)
    return task.output
