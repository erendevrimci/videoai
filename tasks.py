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

celery_app.conf.task_routes = {
    "tasks.create_storyboard_task": {"queue": "storyboard_queue"},
    "tasks.create_final_video_from_storyboard_task": {"queue": "video_queue"},
    "tasks.create_final_video_without_storyboard_task": {"queue": "video_queue"},
}
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

@celery_app.task(name="tasks.create_storyboard_task", bind=True)
def create_storyboard_task(self, project_id: int, caption_id: int, user_id: str, storyboard_name: str, shot_index_size: Optional[int]=None):
    """
    Celery görevi olarak storyboard oluşturan fonksiyon.
    API endpoint'indeki tüm mantık buraya taşındı ve WebSocket ile ilerleme bildirimi eklendi.
    """
    # Gerekli import'lar görev içinde yapılır
    import concurrent.futures
    import os
    import json
    from api.utils.generate_shots_image import generate_images_for_prompts_and_upload_to_supabase
    import video_edit

    task_id = self.request.id
    supabase = get_supabase_client()

    def publish_progress(message: str, stage: str = "PROGRESS"):
        """Helper function to publish progress updates."""
        publish_sync(task_id, json.dumps({
            "status": stage,
            "message": message
        }))

    try:
        shots = []
        publish_progress(f"Storyboard '{storyboard_name}' oluşturma işlemi başladı.", "STARTED")
        logger.info(f"Celery Task [{task_id}]: Creating storyboard for project_id {project_id}")

        # 1. Storyboard'u veritabanına ekle
        storyboard_result = supabase.table("storyboards").insert({
            "name": storyboard_name,
            "user_id": user_id,
            "project_id": project_id,
        }).execute()

        if not storyboard_result.data:
            logger.error(f"Task [{task_id}]: Failed to create storyboard entry in database. Error: {storyboard_result.error}")
            raise Exception("Storyboard database entry failed.")
        
        storyboard_id = storyboard_result.data[0]["id"]
        logger.info(f"Task [{task_id}]: Storyboard entry created with id {storyboard_id}")

        # 2. Video varlıklarını hazırla (Bu, OpenAI çağrısını yapar ve response_json'u kaydeder)
        publish_progress("Yapay zeka ile senaryo analizi ve klip seçimi yapılıyor...")
        success = video_edit.prepare_video_assets(project_id, caption_id, user_id)
        if not success:
            logger.error(f"Task [{task_id}]: Failed to prepare video assets (prepare_video_assets returned False).")
            raise Exception("Failed to prepare video assets.")

        # 3. response_json'u al ve shot'ları oluştur
        publish_progress("Klip dizisi işleniyor ve storyboard çekimleri oluşturuluyor...")
        response_json_result = supabase.table("projects").select("response_json").eq("id", project_id).single().execute()

        if response_json_result.data and response_json_result.data.get("response_json"):
            response_json = response_json_result.data["response_json"]
            try:
                response_json_data = json.loads(response_json)
                if not isinstance(response_json_data, list):
                    logger.warning(f"Task [{task_id}]: response_json for project_id {project_id} is not a list.")
                    response_json_data = []
            except json.JSONDecodeError:
                logger.warning(f"Task [{task_id}]: Invalid JSON in response_json for project_id {project_id}.")
                response_json_data = []

            # --- THREAD POOL KALDIRILDI - Senkron İşlem ---
            # Klipleri tek tek, sırayla işle
            shots_to_insert = []
            for index, clip_data in enumerate(response_json_data):
                publish_progress(f"Processing clip {index + 1}/{len(response_json_data)}: {clip_data.get('clip_name')}", "PROGRESS")
                if not all(key in clip_data for key in ["clip_name", "suggestion", "duration", "explanation", "script_segment", "start_time"]):
                    logger.warning(f"Task [{task_id}]: Missing keys in clip_data for storyboard_id {storyboard_id}. Skipping shot.")
                    continue
                try:
                    # Görevin ana supabase istemcisini kullan
                    signed_url_data = supabase.storage.from_("video-database").create_signed_url(clip_data["clip_name"], 3600)
                    video_url = signed_url_data.get('signedURL') if signed_url_data else None
                    
                    shot = {
                        "approved": False, "video_url": video_url, "duration": clip_data["duration"],
                        "suggestion": clip_data["suggestion"], "explanation": clip_data["explanation"],
                        "clip_name": clip_data["clip_name"], "script_segment": clip_data["script_segment"],
                        "start_time": clip_data["start_time"], "shot_index": index,
                        "storyboard_id": storyboard_id, "user_id": user_id
                    }
                    shots_to_insert.append(shot)
                except Exception as e:
                    logger.error(f"Task [{task_id}]: Error processing clip data {clip_data.get('clip_name')}: {e}")
                    continue
            
            if shots_to_insert:
                publish_progress(f"{len(shots_to_insert)} adet çekim veritabanına kaydediliyor...")
                shot_insert_result = supabase.table("shots").insert(shots_to_insert).execute()
                if shot_insert_result.data:
                    shots = shot_insert_result.data
                    logger.info(f"Task [{task_id}]: Successfully inserted {len(shots)} shots for storyboard {storyboard_id}.")
                else:
                    logger.error(f"Task [{task_id}]: Failed to bulk insert shots for storyboard {storyboard_id}. Error: {shot_insert_result.error}")
            
            prompts = [shot['suggestion'] for shot in shots_to_insert if 'suggestion' in shot]
            if prompts:
                publish_progress(f"{len(prompts)} adet çekim için önizleme görselleri oluşturuluyor...")
                try:
                    generate_images_for_prompts_and_upload_to_supabase(
                        prompts=prompts, user_id=user_id, storyboard_id=storyboard_id, project_id=project_id
                    )
                    logger.info(f"Task [{task_id}]: Image generation tasks for storyboard {storyboard_id} sent successfully.")
                except Exception as img_exc:
                    logger.error(f"Task [{task_id}]: Error initiating image generation for storyboard {storyboard_id}: {img_exc}")

        else:
            logger.warning(f"Task [{task_id}]: No valid response_json found for project {project_id}. Checking shot_index_size.")
            if shot_index_size is not None and shot_index_size > 0:
                publish_progress(f"{shot_index_size} adet boş çekim oluşturuluyor...")
                empty_shots_to_insert = [{
                    "approved": False, "video_url": None, "duration": None,
                    "suggestion": "Empty shot", "explanation": "Automatically generated empty shot",
                    "clip_name": None, "script_segment": None, "start_time": None,
                    "shot_index": i, "storyboard_id": storyboard_id, "user_id": user_id
                } for i in range(shot_index_size)]
                
                if empty_shots_to_insert:
                    shot_insert_result = supabase.table("shots").insert(empty_shots_to_insert).execute()
                    if shot_insert_result.data:
                        shots = shot_insert_result.data
                        logger.info(f"Task [{task_id}]: Inserted {len(shots)} empty shots.")
                    else:
                        logger.error(f"Task [{task_id}]: Failed to insert empty shots. Error: {shot_insert_result.error}")

        # --- Başarı Mesajı ---
        success_message = {
            "status": "SUCCESS",
            "message": "Storyboard created successfully.",
            "result": {
                "storyboard_id": storyboard_id,
            }
        }
        publish_sync(task_id, json.dumps(success_message))
        logger.info(f"✅ Celery Task [{task_id}]: Storyboard creation process finished successfully.")
        return success_message["result"]

    except Exception as e:
        logger.error(f"Celery Task [{task_id}]: Error creating storyboard: {e}", exc_info=True)
        failure_message = {
            "status": "FAILURE",
            "message": str(e)
        }
        publish_sync(task_id, json.dumps(failure_message))
        raise e