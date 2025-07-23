import os
from celery import Celery
from dotenv import load_dotenv
import json
from api.websockets.pubsub import publish_sync  # Güncellendi: Artık senkron sarmalayıcıyı kullanıyoruz
from supabase import create_client
from supabase.client import Client
from celery.utils.log import get_task_logger
from typing import Optional
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

# `api.websockets.pubsub` içindeki `publish_sync` artık doğrudan kullanıldığı için
# bu dosyadaki özel WebSocket publisher fonksiyonlarına gerek kalmadı.
# Bu fonksiyonlar kaldırıldı: _publish_message, get_websocket_publisher, get_redis_client

# Celery uygulamasının yapılandırmasını güncelle
celery_app.conf.update(
    task_track_started=True,
    task_acks_late=True,  # Görevlerin başarıyla tamamlandıktan veya başarısız olduktan sonra onaylanmasını sağlar.
    broker_heartbeat=120,  # Heartbeat aralığını 2 dakikaya çıkarır.
    broker_transport_options={
        # Bir görevin başka bir workera yeniden atanmadan önce ne kadar süre (saniye) görünmez kalacağını belirtir.
        # Uzun video işleme görevleri için bu süreyi artırmak önemlidir.
        'visibility_timeout': 600  # 2 saat
    }
)

# --- Celery Görevleri Logger ---
logger = get_task_logger(__name__)

# --- Lazy Supabase Client Loader ---
_supabase_client: Optional[Client] = None

def get_supabase_client()->Optional[Client]:
    """
    Lazily initializes and returns a singleton Supabase client instance.
    The client is created only on the first call and cached for subsequent calls.
    """
    global _supabase_client
    if _supabase_client is None:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key:
            _supabase_client = create_client(supabase_url, supabase_key)
        else:
            logger.warning("Supabase credentials not found. Signed URL generation will be skipped.")
            # We still set it to something (even None) to avoid re-evaluating env vars.
            _supabase_client = None 
    return _supabase_client


@celery_app.task(name="tasks.create_final_video_from_storyboard_task", bind=True)
def create_final_video_from_storyboard_task(self, storyboard_id: int, project_id: int, user_id: str) -> str:
    """
    Celery görevi olarak video oluşturma işlemini çalıştırır ve WebSocket üzerinden ilerleme bildirir.
    
    Returns:
        Oluşturulan videonun Supabase Storage'daki yolu veya hata mesajı.
    """
    task_id = self.request.id
    
    try:
        # --- Görev Başladı Bildirimi ---
        start_message = {
            "status": "STARTED",
            "message": f"Video generation started for storyboard {storyboard_id}."
        }
        publish_sync(task_id, json.dumps(start_message))
        logger.info(f"Celery task [{task_id}] started.")
        import video_edit
        
        # --- Ana İşlemi Çalıştır (Bu kısım senkron ve engelleyici) ---
        video_path = video_edit.create_video_from_storyboard(
            storyboard_id=storyboard_id,
            project_id=project_id,
            user_id=user_id
        )
        
        # --- Sonucu Bildir ---
        if video_path:
            print(f"Celery task [{task_id}] finished successfully. Video path: {video_path}")
            
            signed_url = None
            supabase = get_supabase_client()
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
            publish_sync(task_id, json.dumps(success_message))
            # Celery'nin kendi sonucuna da URL'i ekleyelim (yedek olarak)
            return success_message["result"]
        else:
            print(f"Celery task [{task_id}] finished with failure.")
            failure_message = {
                "status": "FAILURE",
                "message": "Video generation failed in the editing process."
            }
            publish_sync(task_id, json.dumps(failure_message))
            return "Video generation failed."

    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {e}"
        print(f"Celery task [{task_id}] encountered an exception: {error_message}\n{traceback.format_exc()}")
        
        failure_message = {
            "status": "FAILURE",
            "message": error_message
        }
        # Hata durumunda da mesajı yayınlamaya çalış
        publish_sync(task_id, json.dumps(failure_message))
        # Celery'nin hatayı düzgün işlemesi için yeniden fırlat
        raise e

@celery_app.task(name="tasks.create_final_video_without_storyboard_task", bind=True)
def create_final_video_without_storyboard_task(self, project_id: int, caption_id: int, user_id: str) -> str:
    """
    Creates a final video directly from voiceover and captions without a storyboard.
    1. Prepares assets and generates a clip sequence using AI.
    2. Produces the final video with voice, music, and subtitles.
    """
    import gc
    import video_edit
    
    task_id = self.request.id
    
    try:
        # 1. Görevin başladığını bildir
        publish_sync(task_id, json.dumps({
            "status": "STARTED",
            "message": "Video generation process has started."
        }))
        
        # 2. Varlıkları hazırla (AI klip seçimi)
        publish_sync(task_id, json.dumps({
            "status": "PROGRESS",
            "message": "Preparing assets and generating clip sequence with AI..."
        }))
        
        assets_prepared = video_edit.prepare_video_assets(
            project_id=project_id,
            caption_id=caption_id,
            user_id=user_id
        )
        gc.collect()

        if not assets_prepared:
            raise Exception("Failed to prepare video assets. The process was stopped.")

        publish_sync(task_id, json.dumps({
            "status": "PROGRESS",
            "message": "Asset preparation complete. Starting final video production..."
        }))
        
        # 3. Nihai videoyu üret
        video_path = video_edit.produce_final_video(
            project_id=project_id,
            caption_id=caption_id,
            user_id=user_id,
            timeline_mode=False
        )
        gc.collect()

        if not video_path:
            raise Exception("Final video production failed.")
            
        # 4. Başarı mesajı gönder
        signed_url = None
        supabase = get_supabase_client()
        if supabase:
            try:
                signed_url_response = supabase.storage.from_("final-videos").create_signed_url(video_path, 3600)
                signed_url = signed_url_response.get("signedURL")
            except Exception as e_sign:
                logger.error(f"Could not create signed URL for {video_path}: {e_sign}")
        
        success_message = {
            "status": "SUCCESS",
            "message": "Video generation completed successfully.",
            "result": {
                "video_path": video_path,
                "video_url": signed_url
            }
        }
        publish_sync(task_id, json.dumps(success_message))
        return success_message["result"]

    except Exception as e:
        logger.error(f"Task {task_id} (without storyboard) failed: {e}", exc_info=True)
        failure_message = {
            "status": "FAILURE",
            "message": str(e)
        }
        publish_sync(task_id, json.dumps(failure_message))
        raise 

