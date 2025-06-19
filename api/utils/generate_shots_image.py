import requests
import os
import json
import concurrent.futures
import uuid
import io
import mimetypes
import base64
from supabase import create_client, Client
from dotenv import load_dotenv

load_dotenv()

# Tek bir prompt için görselleri getiren, Supabase'e yükleyen ve DB'ye kaydeden yardımcı fonksiyon
def _process_single_prompt_for_supabase(
    prompt_index_tuple: tuple[int, str], # (index, prompt_text)
    openai_api_key: str,
    n_images_per_prompt: int,
    image_size: str,
    user_id: str | None,
    storyboard_id: int | None,
    batch_id: str
) -> list[dict] | None:
    """Tek bir prompt için OpenAI API'sinden görselleri alır, Supabase'e yükler ve DB'ye kaydeder."""
    # Her iş parçacığı (thread) için ayrı bir Supabase istemcisi oluşturulur.
    # Bu, "Server disconnected" gibi bağlantı hatalarını önler.
    supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
    
    prompt_index, prompt_text = prompt_index_tuple
    openai_url = "https://api.openai.com/v1/images/generations"
    
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }
    
    payload: dict[str, any] = {
        "model": "gpt-image-1",
        "prompt": prompt_text,
        "n": n_images_per_prompt,
        "size": image_size,
        "quality": "low",
    }
    
    openai_response_obj = None

    try:
        openai_response_obj = requests.post(openai_url, headers=headers, json=payload)
        
        openai_response_obj.raise_for_status()
        response_json = openai_response_obj.json()
        data_list = response_json.get('data', []) 
        
        actual_b64_strings = []
        if isinstance(data_list, list):
            for item in data_list:
                if isinstance(item, dict) and 'b64_json' in item:
                    actual_b64_strings.append(item['b64_json'])
        
        if not actual_b64_strings:
            print(f"Hata (prompt: '{prompt_text}'): OpenAI API'den 'b64_json' formatında görsel verisi ayıklanamadı. Yanıt: {response_json}")
            return None
            
    except requests.exceptions.RequestException as e:
        content = openai_response_obj.text if openai_response_obj else "N/A"
        print(f"OpenAI API isteği sırasında bir hata oluştu (prompt: '{prompt_text}'): {e}. Yanıt: {content}")
        return None
    except json.JSONDecodeError:
        content = openai_response_obj.text if openai_response_obj else "N/A"
        print(f"Hata (prompt: '{prompt_text}'): OpenAI API yanıtı JSON formatında değil. Yanıt: {content}")
        return None

    # Görselleri indir, Supabase'e yükle ve DB'ye kaydet
    uploaded_image_records = []
    for img_idx, b64_encoded_image in enumerate(actual_b64_strings):
        try:
            # 1. Görseli base64'ten çöz
            image_bytes = base64.b64decode(b64_encoded_image)
            
            # İçerik tipini ve uzantıyı PNG olarak varsayalım (DALL-E b64_json için yaygın)
            content_type = 'image/png'
            extension = ".png"
            
            # 2. Supabase Storage'a yükle
            file_name = f"{batch_id}/{prompt_index}_{img_idx}{extension}"
            storage_path = f"images/{file_name}"
            
            file_options = {"content-type": content_type, "cache-control": "3600", "upsert": "false"}

            supabase.storage.from_("videos").upload(
                path=storage_path,
                file=image_bytes,
                file_options=file_options
            )
            
            public_url_data = supabase.storage.from_("videos").get_public_url(storage_path)
            public_url = public_url_data if isinstance(public_url_data, str) else public_url_data.get('publicUrl')

            if not public_url:
                 print(f"Hata (prompt: '{prompt_text}', img_idx: {img_idx}): Supabase Storage'dan public URL alınamadı.")
                 continue

            # 3. Supabase DB'ye kaydet
            image_record_to_insert = {
                "id": str(uuid.uuid4()),
                "url": public_url,
                "prompt": prompt_text,
                "model": "gpt-image-1",
                "size": image_size,
                "quality": "low",
                "user_id": user_id,
                "storyboard_id": storyboard_id,
                "batch_id": batch_id,
                "shot_index": prompt_index
            }
            
            insert_response = supabase.table("images").insert(image_record_to_insert).execute()
            
            if hasattr(insert_response, 'error') and insert_response.error:
                print(f"Supabase DB'ye kayıt sırasında hata (prompt: '{prompt_text}', url: {public_url}): {insert_response.error}")
                continue
            
            if hasattr(insert_response, 'data') and insert_response.data:
                 uploaded_image_records.append(insert_response.data[0])
            else:
                 uploaded_image_records.append(image_record_to_insert)

        except base64.binascii.Error as e:
             print(f"Base64 decode hatası (prompt: '{prompt_text}', img_idx: {img_idx}): {e}")
             continue
        except Exception as e:
            print(f"Supabase'e yükleme/kayıt veya b64 decode sırasında hata (prompt: '{prompt_text}', img_idx: {img_idx}): {e}")
            continue
            
    return uploaded_image_records if uploaded_image_records else None


def generate_images_for_prompts_and_upload_to_supabase(
    prompts: list[str],
    user_id: str | None, 
    storyboard_id: int | None, 
    n_images_per_prompt: int = 1,
    image_size: str = "1024x1024",
    openai_model: str = "dall-e-3",
) -> dict[str, list[dict] | None]:
    """
    OpenAI DALL-E API'sini kullanarak verilen bir metin listesinden paralel olarak görseller oluşturur (b64_json formatında),
    Supabase Storage'a yükler ve bilgilerini Supabase veritabanına kaydeder.

    Args:
        prompts: Görsel oluşturmak için kullanılacak metinlerin listesi.
        user_id: Görseli oluşturan kullanıcının ID'si.
        storyboard_id: Görselin ait olduğu storyboard ID'si.
        n_images_per_prompt: Her bir prompt için oluşturulacak görsel sayısı.
        image_size: Oluşturulacak görselin boyutu. DALL-E 3: "1024x1024", "1024x1792", "1792x1024".
        openai_model: Kullanılacak OpenAI modeli. Not: Bu parametre şu anda _process_single_prompt_for_supabase içinde sabitlenmiştir.
                      Eğer modelin dinamik olması isteniyorsa kodun güncellenmesi gerekir.

    Returns:
        Her bir prompt'u anahtar olarak ve Supabase'e kaydedilen görsel kayıtlarının
        (veya o prompt için hata durumunda None) listesini değer olarak içeren bir sözlük.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    
    if not openai_api_key:
        print("Hata: OPENAI_API_KEY ortam değişkeni bulunamadı.")
        return {prompt: None for prompt in prompts}
   
    results: dict[str, list[dict] | None] = {}
    batch_id = str(uuid.uuid4()) 
    
    max_workers = 2

    indexed_prompts = list(enumerate(prompts))

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_prompt_text = {
            executor.submit(
                _process_single_prompt_for_supabase,
                prompt_idx_tuple, 
                openai_api_key,
                n_images_per_prompt,
                image_size,
                user_id,
                storyboard_id,
                batch_id
            ): prompt_idx_tuple[1]
            for prompt_idx_tuple in indexed_prompts
        }
        
        for future in concurrent.futures.as_completed(future_to_prompt_text):
            original_prompt_text = future_to_prompt_text[future]
            try:
                image_data_list = future.result() 
                results[original_prompt_text] = image_data_list
            except Exception as exc:
                print(f"'{original_prompt_text}' için ana iş parçacığında bir istisna oluştu: {exc}")
                results[original_prompt_text] = None
                
    return results

