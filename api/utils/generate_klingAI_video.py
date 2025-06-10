import replicate
import os
import aiohttp
from typing import Optional


api_key = os.getenv("REPLICATE_API_KEY")
replicate_client = replicate.Client(api_token=api_key)
if not api_key:
    # Bu, sunucu başladığında bir hata fırlatır ve sorunu hemen belli eder.
    raise ValueError("REPLICATE_API_KEY ortam değişkeni bulunamadı veya boş. Lütfen .env dosyanızı veya sunucu yapılandırmanızı kontrol edin.")

async def generate_klingAI_video(prompt: str, negativePrompt: str, duration: int, cfgScale: float, aspectRatio: str, startImage: str, endImage: Optional[str] = None):
    
    print(f"KlingAI | Segment '{prompt[:30]}...' için üretim başlıyor.")
    effective_cfgScale = cfgScale if cfgScale is not None else 0.5
    
    input_params = {
        "prompt": prompt,
        "duration": duration if duration else 5,
        "cfg_scale": effective_cfgScale,
        "aspect_ratio": aspectRatio if aspectRatio else "9:16",
        "start_image": startImage if startImage else None,
    }

    if endImage is not None:
        input_params["end_image"] = endImage
    
    if negativePrompt is not None:
        input_params["negative_prompt"] = negativePrompt
    else:
        input_params["negative_prompt"] = ""
    
    print(f"KlingAI | Replicate API çağrılıyor...")
    raw_prediction_output = await replicate_client.async_run(
        "kwaivgi/kling-v1.6-pro",
        input=input_params,
    )
    print(f"KlingAI | Replicate API'den yanıt alındı.")
    print(f"DEBUG: Yanıt tipi: {type(raw_prediction_output)}, Yanıt içeriği: {raw_prediction_output}")

    # Gelen yanıtı (FileOutput nesnesi dahil) doğrudan string'e çevirerek URL'yi alalım.
    # Bu, en sağlam yöntemdir.
    video_url_to_download = str(raw_prediction_output).strip()

    if not video_url_to_download or not video_url_to_download.startswith('http'):
        print(f"HATA: KlingAI | Geçerli bir URL alınamadı. Ham Çıktı: {raw_prediction_output}")
        raise ValueError("Could not extract a valid video URL from Replicate output.")

    print(f"KlingAI | Video URL'si indiriliyor: {video_url_to_download}")
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(video_url_to_download, timeout=120) as response:
                response.raise_for_status()
                contents = await response.read()
                print(f"KlingAI | Video başarıyla indirildi. Boyut: {len(contents)} bytes.")
                return contents
    except Exception as e:
        print(f"HATA: KlingAI | Video indirme hatası: {e}")
        raise e
