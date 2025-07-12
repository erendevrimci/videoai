import os
from celery import Celery
import video_edit
from dotenv import load_dotenv

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

celery_app.conf.update(
    task_track_started=True,
)

@celery_app.task(name="tasks.create_final_video_task")
def create_final_video_task(storyboard_id: int, project_id: int, user_id: str) -> str:
    """
    Celery görevi olarak video oluşturma işlemini çalıştırır.
    
    Returns:
        Oluşturulan videonun Supabase Storage'daki yolu veya hata mesajı.
    """
    try:
        print(f"Celery task started for storyboard_id: {storyboard_id}")
        video_path = video_edit.create_video_from_storyboard(
            storyboard_id=storyboard_id,
            project_id=project_id,
            user_id=user_id
        )
        if video_path:
            print(f"Celery task finished successfully. Video path: {video_path}")
            return video_path
        else:
            print(f"Celery task finished with failure: video_edit.create_video_from_storyboard returned None.")
            return "Video generation failed."
    except Exception as e:
        import traceback
        print(f"An exception occurred in Celery task: {e}")
        traceback.print_exc()
        # Celery'nin hatayı düzgün işlemesi için yeniden fırlat
        raise e 