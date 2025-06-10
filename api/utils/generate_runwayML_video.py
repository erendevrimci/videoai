from runwayml import AsyncRunwayML
import asyncio
import os
from typing import Optional
import aiohttp
import base64

async def url_to_base64(url: str) -> Optional[str]:
    """
    Asenkron olarak bir URL'den resmi indirir ve base64'e dönüştürür.
    Hata durumunda None döndürür.
    """
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=15) as response:
                response.raise_for_status()
                content = await response.read()
                return base64.b64encode(content).decode('utf-8')
    except Exception as e:
        print(f"Error converting URL to base64 async: {e}")
        return None

async def generate_runwayML_video(prompt: str, duration: int, aspectRatio: str, startImage: str, endImage: Optional[str]):
    api_key = os.getenv("RUNWAYML_API_SECRET")
    client = AsyncRunwayML(api_key=api_key)
    
    start_image_base64 = await url_to_base64(startImage)
    if not start_image_base64:
        raise ValueError("Başlangıç resmi URL'si dönüştürülemedi veya geçersiz.")

    prompt_image_data = f"data:image/png;base64,{start_image_base64}"
    
    if endImage:
        end_image_base64 = await url_to_base64(endImage)
        if not end_image_base64:
            raise ValueError("Bitiş resmi URL'si dönüştürülemedi veya geçersiz.")
            
        prompt_image_data = [
            {
                "uri": f"data:image/png;base64,{start_image_base64}",
                "position": "first"
            },
            {
                "uri": f"data:image/png;base64,{end_image_base64}",
                "position": "last"
            }
        ]
    adjusted_aspect_ratio = None
    match aspectRatio:
        case "5:3":
            adjusted_aspect_ratio = "1280:768"
        case "3:5":
            adjusted_aspect_ratio = "768:1280"
        
    task = await client.image_to_video.create(
        model="gen3a_turbo",
        prompt_text=prompt,
        duration=duration,
        ratio=adjusted_aspect_ratio,
        prompt_image=prompt_image_data,
    )
    task_id = task.id
    await asyncio.sleep(10)

    task = await client.tasks.retrieve(task_id)

    while task.status not in ['SUCCEEDED', 'FAILED']:
        await asyncio.sleep(10)
        task = await client.tasks.retrieve(task_id)
    return task.output
