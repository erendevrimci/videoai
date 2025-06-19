import os
from celery import Celery
from dotenv import load_dotenv
import video_edit # Asıl işi yapacak modülümüzü import ediyoruz

# .env dosyasını yükle (özellikle yerel geliştirme için)
load_dotenv()

# Redis URL'sini ortam değişkenlerinden alıyoruz. Railway bunu otomatik olarak sağlayacak.
# Yerel geliştirme için varsayılan bir değer belirliyoruz.
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery uygulamasını oluşturuyoruz
# 'tasks' ana modülün adıdır.
# broker: Görevlerin gönderileceği mesaj aracısı (Redis).
# backend: Görev sonuçlarının saklanacağı yer (yine Redis kullanabiliriz).
celery_app = Celery(
    'tasks',
    broker=redis_url,
    backend=redis_url
)

# Bu decorator, aşağıdaki fonksiyonu bir Celery görevi olarak işaretler.
@celery_app.task
def create_final_video_task(storyboard_id: int, project_id: int, user_id: str):
    """
    Celery tarafından arka planda çalıştırılacak olan video oluşturma görevi.
    """
    print(f"Celery task started: Creating video for storyboard_id={storyboard_id}")
    try:
        # Daha önce API endpoint'i içinde doğrudan çağrılan fonksiyonu burada çağırıyoruz.
        video_storage_path = video_edit.create_video_from_storyboard(
            storyboard_id=storyboard_id,
            project_id=project_id,
            user_id=user_id
        )
        
        if video_storage_path:
            print(f"Celery task finished: Video created at {video_storage_path}")
            return {"status": "Success", "path": video_storage_path}
        else:
            print("Celery task failed: create_video_from_storyboard returned None")
            return {"status": "Failure", "error": "Video creation process failed."}
            
    except Exception as e:
        import traceback
        print(f"Celery task encountered an exception: {e}\n{traceback.format_exc()}")
        # Hata durumunda yeniden deneme mekanizmaları da eklenebilir.
        return {"status": "Error", "error": str(e)} 