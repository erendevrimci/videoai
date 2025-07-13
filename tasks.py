import os
from celery import Celery
from dotenv import load_dotenv
import json
import asyncio
import gc  # Garbage collection için

# Heavy imports'ları lazy loading'e çevir
# from api.websockets.pubsub import publish_message  # KALDIRILDI
# from supabase import create_client  # KALDIRILDI

# .env dosyasını yükle
load_dotenv()

# Redis URL'sini ortam değişkeninden al, yoksa varsayılan bir değer kullan
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Celery uygulamasını oluştur
celery_app = Celery(
    "tasks",
    broker=REDIS_URL,
    backend=REDIS_URL,
    broker_connection_retry_on_startup=True
)

# Memory-optimized configuration
celery_app.conf.update(
    task_track_started=True,
    # Worker concurrency ayarları
    worker_concurrency=2,  # Her worker sadece 1 task
    worker_prefetch_multiplier=1,  # Prefetch sadece 1 task
    worker_max_tasks_per_child=2,  # 2 task sonra restart (memory leak prevention)
    # Task timeout
    task_soft_time_limit=1800,  # 30 dakika
    task_time_limit=2400,  # 40 dakika
    # Result cleanup
    result_expires=3600,  # 1 saat
    # Memory optimization
    worker_hijack_root_logger=False,
    worker_log_color=False,
    # Import optimization
    worker_enable_remote_control=False,
    worker_send_task_events=False,
)

# Lazy loading helper functions
def get_supabase_client():
    """Lazy load Supabase client"""
    if not hasattr(get_supabase_client, '_client'):
        from supabase import create_client
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        if supabase_url and supabase_key:
            get_supabase_client._client = create_client(supabase_url, supabase_key)
        else:
            get_supabase_client._client = None
    return get_supabase_client._client

def get_websocket_publisher():
    """Lazy load websocket publisher"""
    if not hasattr(get_websocket_publisher, '_publisher'):
        from api.websockets.pubsub import publish_message
        get_websocket_publisher._publisher = publish_message
    return get_websocket_publisher._publisher

@celery_app.task(name="tasks.create_final_video_task", bind=True)
def create_final_video_task(self, storyboard_id: int, project_id: int, user_id: str) -> str:
    """
    Memory-optimized video creation task with lazy loading
    """
    import gc
    
    # Lazy load heavy modules only when needed
    import video_edit
    
    # Lazy load clients
    supabase = get_supabase_client()
    publish_message = get_websocket_publisher()
    
    task_id = self.request.id
    loop = asyncio.get_event_loop()

    try:
        # Start message
        start_message = {
            "status": "STARTED",
            "message": f"Video generation started for storyboard {storyboard_id}."
        }
        loop.run_until_complete(publish_message(task_id, json.dumps(start_message)))
        
        # Force garbage collection before heavy processing
        gc.collect()
        
        # Main processing
        video_path = video_edit.create_video_from_storyboard(
            storyboard_id=storyboard_id,
            project_id=project_id,
            user_id=user_id
        )
        
        # Force garbage collection after processing
        gc.collect()
        
        # Success handling
        if video_path:
            signed_url = None
            if supabase:
                try:
                    signed_url_response = supabase.storage.from_("final-videos").create_signed_url(video_path, 3600)
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
            loop.run_until_complete(publish_message(task_id, json.dumps(success_message)))
            return success_message["result"]
        else:
            failure_message = {
                "status": "FAILURE",
                "message": "Video generation failed in the editing process."
            }
            loop.run_until_complete(publish_message(task_id, json.dumps(failure_message)))
            return "Video generation failed."

    except Exception as e:
        import traceback
        error_message = f"An unexpected error occurred: {e}"
        print(f"Celery task [{task_id}] encountered an exception: {error_message}\n{traceback.format_exc()}")
        
        failure_message = {
            "status": "FAILURE",
            "message": error_message
        }
        loop.run_until_complete(publish_message(task_id, json.dumps(failure_message)))
        raise e
    finally:
        # Cleanup after task completion
        gc.collect() 