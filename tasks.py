import os
from celery import Celery
import video_edit
from dotenv import load_dotenv
import json
import asyncio
from api.websockets.pubsub import publish_message # Publish fonksiyonumuzu import ediyoruz
from supabase import create_client

# .env dosyasını yükle
load_dotenv()

# Redis URL'sini ortam değişkeninden al, yoksa varsayılan bir değer kullan
# Railway genellikle REDIS_URL gibi bir değişken sağlar.
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery uygulamasını oluştur
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL, # Sonuçları da Redis'te saklamak için
    broker_connection_retry_on_startup=True
)
# --- Supabase istemcisini burada da oluşturalım ---
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_KEY")
if supabase_url and supabase_key:
    supabase = create_client(supabase_url, supabase_key)
else:
    print("Warning: Supabase credentials not found. Signed URL generation will be skipped in tasks.")
    supabase = None
# --- Bitiş ---

celery_app.conf.update(
    task_track_started=True,
)

@celery_app.task(name="tasks.create_final_video_task", bind=True)
def create_final_video_task(self, storyboard_id: int, project_id: int, user_id: str) -> str:
    """
    Celery görevi olarak video oluşturma işlemini çalıştırır ve WebSocket üzerinden ilerleme bildirir.
    
    Returns:
        Oluşturulan videonun Supabase Storage'daki yolu veya hata mesajı.
    """
    task_id = self.request.id
    
    async def run_and_publish():
        try:
            # --- Görev Başladı Bildirimi ---
            start_message = {
                "status": "STARTED",
                "message": f"Video generation started for storyboard {storyboard_id}."
            }
            await publish_message(task_id, json.dumps(start_message))
            print(f"Celery task [{task_id}] started.")

            # --- Ana İşlemi Çalıştır ---
            video_path = video_edit.create_video_from_storyboard(
                storyboard_id=storyboard_id,
                project_id=project_id,
                user_id=user_id
            )
            
            # --- Sonucu Bildir ---
            if video_path:
                print(f"Celery task [{task_id}] finished successfully. Video path: {video_path}")
                
                signed_url = None
                if supabase:
                    try:
                        # Kullanıcının videoya erişebilmesi için imzalı bir URL oluştur
                        signed_url_response = supabase.storage.from_("final-videos").create_signed_url(video_path, 3600) # 1 saat geçerli
                        signed_url = signed_url_response.get("signedURL")
                    except Exception as e_sign:
                        print(f"Could not create signed URL for {video_path}: {e_sign}")

                success_message = {
                    "status": "SUCCESS",
                    "message": "Video generation successful.",
                    "result": {
                        "video_path": video_path,
                        "video_url": signed_url
                    }
                }
                await publish_message(task_id, json.dumps(success_message))
                # Celery'nin kendi sonucuna da URL'i ekleyelim (yedek olarak)
                return success_message["result"]
            else:
                print(f"Celery task [{task_id}] finished with failure.")
                failure_message = {
                    "status": "FAILURE",
                    "message": "Video generation failed in the editing process."
                }
                await publish_message(task_id, json.dumps(failure_message))
                return "Video generation failed."

        except Exception as e:
            import traceback
            error_message = f"An unexpected error occurred: {e}"
            print(f"Celery task [{task_id}] encountered an exception: {error_message}\n{traceback.format_exc()}")
            
            failure_message = {
                "status": "FAILURE",
                "message": error_message
            }
            await publish_message(task_id, json.dumps(failure_message))
            # Celery'nin hatayı düzgün işlemesi için yeniden fırlat
            raise e

    # Celery senkron bir ortamda çalıştığı için, asenkron publish fonksiyonumuzu
    # çalıştırmak için asyncio.run kullanıyoruz.
    return asyncio.run(run_and_publish()) 