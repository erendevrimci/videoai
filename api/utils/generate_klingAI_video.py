import replicate
import os
import requests 
from typing import Optional


api_key = os.getenv("REPLICATE_API_KEY")
replicate_client = replicate.Client(api_token=api_key)

def generate_klingAI_video(prompt: str, negativePrompt: str, duration: int, cfgScale: float, aspectRatio: str, startImage: str, endImage: Optional[str] = None):
    
    effective_cfgScale = cfgScale if cfgScale is not None else 0.5
    raw_prediction_output = None
    
    input_params = {
        "prompt": prompt,
        "duration": duration if duration else 5,
        "cfg_scale": effective_cfgScale,
        "aspect_ratio": aspectRatio if aspectRatio else "9:16",
        "start_image": startImage if startImage else None,
    }

    if endImage is not None:
        input_params["end_image"] = endImage
        print(f"Generating video with start and end image. Params: {input_params}")
    if negativePrompt is not None:
        input_params["negative_prompt"] = negativePrompt
    else:
        input_params["negative_prompt"] = ""

    raw_prediction_output = replicate_client.run(
        "kwaivgi/kling-v1.6-pro",
        input=input_params
    )

    

    video_url_to_download = None
    if isinstance(raw_prediction_output, str): 
        video_url_to_download = raw_prediction_output
    elif isinstance(raw_prediction_output, list) and len(raw_prediction_output) > 0:
        first_item = raw_prediction_output[0]
        if isinstance(first_item, str):
            video_url_to_download = first_item
        elif hasattr(first_item, 'url') and isinstance(first_item.url, str):
            video_url_to_download = first_item.url
            print(f"Extracted URL from list item object: {video_url_to_download}")
        else:
            print(f"List item is not a recognized URL string or an object with a .url attribute: {type(first_item)}")
    elif hasattr(raw_prediction_output, 'url') and isinstance(raw_prediction_output.url, str):
        video_url_to_download = raw_prediction_output.url
        print(f"Extracted URL from direct object: {video_url_to_download}")
    
    if not video_url_to_download or not video_url_to_download.startswith('http'):
        print(f"Error: Could not extract a valid HTTP/HTTPS URL from Replicate output. Extracted: '{video_url_to_download}'. Raw Output was: {raw_prediction_output}")
        raise ValueError("Could not extract a valid video URL from Replicate output.")

    print(f"Attempting to download video from URL: {video_url_to_download}")
    
    try:
        # Modern ve daha sağlam bir yaklaşım için requests kütüphanesini kullanalım
        response = requests.get(video_url_to_download, timeout=120) # Timeout süresini artırdım (120 saniye)
        response.raise_for_status()  # HTTP 4xx veya 5xx hataları için exception fırlatır
        contents = response.content
        print(f"Successfully downloaded {len(contents)} bytes using requests.")
        return contents
    except requests.exceptions.RequestException as e:
        print(f"Error downloading video using requests from {video_url_to_download}: {e}")
        # Orijinal urllib.request ile bir deneme daha yapılabilir (opsiyonel fallback)
        # try:
        #     print(f"Retrying download with urllib.request for URL: {video_url_to_download}")
        #     contents = urllib.request.urlopen(video_url_to_download).read()
        #     print(f"Successfully downloaded {len(contents)} bytes using urllib.request on retry.")
        #     return contents
        # except Exception as urle:
        #     print(f"Error downloading video using urllib.request on retry from {video_url_to_download}: {urle}")
        #     raise urle # İkinci deneme de başarısız olursa hatayı yükselt
        raise e # requests ile olan ilk hatayı yükselt
