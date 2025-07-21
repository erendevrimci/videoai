import os
import shutil
import tempfile
import zipfile
import logging
import json
from supabase import Client

# Log yapılandırması
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _get_project_data(supabase: Client, project_id: int):
    """Proje verilerini çeker."""
    try:
        return supabase.table("projects").select("name").eq("id", project_id).single().execute().data
    except Exception as e:
        logger.error(f"Proje verileri alınırken hata: {e}")
        return None

def _download_and_save_file(supabase: Client, bucket_name: str, file_path: str, local_dir: str):
    """Supabase storage'dan dosya indirir ve yerel dizine kaydeder."""
    if not file_path:
        logger.warning("Dosya yolu boş, indirme atlanıyor.")
        return False
    try:
        os.makedirs(local_dir, exist_ok=True)
        local_path = os.path.join(local_dir, os.path.basename(file_path))
        
        response = supabase.storage.from_(bucket_name).download(file_path)
        
        if response:
            with open(local_path, "wb") as f:
                f.write(response)
            logger.info(f"{file_path} dosyası {local_path} konumuna indirildi.")
            return True
        else:
            logger.warning(f"{bucket_name} bucket'ından {file_path} indirilemedi.")
            return False
            
    except Exception as e:
        # Hata mesajında daha fazla detay verelim
        logger.error(f"Dosya indirilirken hata (bucket: {bucket_name}, path: {file_path}): {e}", exc_info=True)
        return False

def _fetch_scripts(supabase: Client, project_id: int, target_dir: str):
    """Projedeki scriptleri çeker ve kaydeder."""
    try:
        scripts_dir = os.path.join(target_dir, "scripts")
        os.makedirs(scripts_dir, exist_ok=True)
        scripts = supabase.table("scripts").select("id, script, topic").eq("project_id", project_id).execute().data
        for script in scripts:
            file_name = f"script_{script['id']}_{script.get('topic', 'untitled')}.txt"
            with open(os.path.join(scripts_dir, file_name), "w", encoding="utf-8") as f:
                f.write(script.get('script', ''))
        logger.info(f"{len(scripts)} adet script dosyası kaydedildi.")
    except Exception as e:
        logger.error(f"Scriptler alınırken hata: {e}")

def _fetch_voiceovers(supabase: Client, project_id: int, target_dir: str):
    """Projedeki seslendirmeleri çeker ve indirir."""
    try:
        voiceovers_dir = os.path.join(target_dir, "voiceovers")
        voiceovers = supabase.table("voice_over").select("voice_name").eq("project_id", project_id).execute().data
        for vo in voiceovers:
            _download_and_save_file(supabase, "voice-over-files", vo["voice_name"], voiceovers_dir)
        logger.info(f"{len(voiceovers)} adet seslendirme dosyası işlendi.")
    except Exception as e:
        logger.error(f"Seslendirmeler alınırken hata: {e}")

def _fetch_captions(supabase: Client, project_id: int, target_dir: str):
    """Projedeki altyazıları çeker ve json olarak kaydeder."""
    try:
        captions_dir = os.path.join(target_dir, "captions")
        os.makedirs(captions_dir, exist_ok=True)
        captions = supabase.table("captions").select("id, caption_segments, caption_json").eq("project_id", project_id).execute().data
        for caption in captions:
            file_name = f"caption_{caption['id']}.json"
            content = caption.get('caption_segments') or caption.get('caption_json') or {}
            with open(os.path.join(captions_dir, file_name), "w", encoding="utf-8") as f:
                json.dump(content, f, ensure_ascii=False, indent=4)
        logger.info(f"{len(captions)} adet altyazı dosyası kaydedildi.")
    except Exception as e:
        logger.error(f"Altyazılar alınırken hata: {e}")


def _fetch_images(supabase: Client, project_id: int, target_dir: str):
    """Projedeki görselleri çeker ve indirir."""
    try:
        images_dir = os.path.join(target_dir, "images")
        images = supabase.table("images").select("url").eq("project_id", project_id).execute().data
        for img in images:
            # `upload_image` fonksiyonuna göre bucket adı 'videos'
            _download_and_save_file(supabase, "videos", img["url"], images_dir)
        logger.info(f"{len(images)} adet görsel dosyası işlendi.")
    except Exception as e:
        logger.error(f"Görseller alınırken hata: {e}")

def _fetch_generated_videos(supabase: Client, project_id: int, target_dir: str):
    """Projeye ait oluşturulmuş videoları çeker ve indirir."""
    try:
        gen_videos_dir = os.path.join(target_dir, "generated_videos")
        videos = supabase.table("generated_videos").select("path, name").eq("project_id", project_id).execute().data
        for video in videos:
            # path veya name sütunu dosya yolunu tutabilir, ikisini de kontrol et
            video_path = video.get("path") or video.get("name")
            _download_and_save_file(supabase, "videos", video_path, gen_videos_dir)
        logger.info(f"{len(videos)} adet oluşturulmuş video dosyası işlendi.")
    except Exception as e:
        logger.error(f"Oluşturulmuş videolar alınırken hata: {e}")


def _fetch_final_videos(supabase: Client, project_id: int, target_dir: str):
    """Projenin final videolarını çeker ve indirir."""
    try:
        final_videos_dir = os.path.join(target_dir, "final_videos")
        videos = supabase.table("final_videos").select("video_name").eq("project_id", project_id).execute().data
        for video in videos:
            _download_and_save_file(supabase, "final-videos", video["video_name"], final_videos_dir)
        logger.info(f"{len(videos)} adet final video dosyası işlendi.")
    except Exception as e:
        logger.error(f"Final videolar alınırken hata: {e}")


def export_project_assets(project_id: int, user_id: str, supabase: Client):
    """
    Bir projenin tüm varlıklarını (scriptler, seslendirmeler, vb.) toplar,
    bir zip dosyası oluşturur, Supabase'e yükler ve imzalı bir URL döndürür.
    """
    # 1. Export kaydı oluştur
    try:
        export_record = supabase.table("project_exports").insert({
            "project_id": project_id,
            "user_id": user_id,
            "status": "pending"
        }).execute().data[0]
        export_id = export_record['id']
    except Exception as e:
        logger.error(f"Export kaydı oluşturulamadı: {e}")
        return {"success": False, "message": "Dışa aktarma başlatılamadı."}

    try:
        project_data = _get_project_data(supabase, project_id)
        if not project_data:
            raise Exception("Proje bulunamadı.")

        project_name = project_data.get("name", "export")
        
        with tempfile.TemporaryDirectory() as temp_dir:
            project_folder_name = f"project_{project_id}_{project_name}".replace(" ", "_")
            export_base_path = os.path.join(temp_dir, project_folder_name)
            os.makedirs(export_base_path, exist_ok=True)

            logger.info(f"'{project_name}' projesi için dışa aktarma başlatıldı...")

            # Varlıkları al ve kaydet
            _fetch_scripts(supabase, project_id, export_base_path)
            _fetch_voiceovers(supabase, project_id, export_base_path)
            _fetch_captions(supabase, project_id, export_base_path)
            _fetch_images(supabase, project_id, export_base_path)
            _fetch_generated_videos(supabase, project_id, export_base_path)
            _fetch_final_videos(supabase, project_id, export_base_path)
            
            # Geçici proje klasörünü ZIP'le
            zip_file_name = f"{project_folder_name}.zip"
            zip_file_path = os.path.join(temp_dir, zip_file_name)
            shutil.make_archive(os.path.join(temp_dir, project_folder_name), 'zip', temp_dir, project_folder_name)
            logger.info(f"Proje varlıkları {zip_file_path} olarak arşivlendi.")
            
            # ZIP dosyasını Supabase'e yükle
            bucket_name = "export-files"
            remote_zip_path = f"{user_id}/{zip_file_name}"

            with open(zip_file_path, "rb") as f:
                file_size = os.path.getsize(zip_file_path)
                supabase.storage.from_(bucket_name).upload(
                    path=remote_zip_path,
                    file=f,
                    file_options={"content-type": "application/zip", "upsert": "true"}
                )
            logger.info(f"Zip dosyası {bucket_name} bucket'ına yüklendi: {remote_zip_path}")
            
            # İmzalı URL oluştur
            signed_url_response = supabase.storage.from_(bucket_name).create_signed_url(remote_zip_path, 3600)
            logger.info(f"İmzalı URL oluşturuldu: {signed_url_response}")
            if not signed_url_response or not signed_url_response.get("signedURL"):
                 raise Exception("Arşiv yüklendi ancak URL oluşturulamadı.")

            # Export kaydını güncelle
            supabase.table("project_exports").update({
                "status": "completed",
                "file_path": remote_zip_path,
                "file_size": file_size
            }).eq("id", export_id).execute()

            return {
                "success": True, 
                "message": "Proje başarıyla dışa aktarıldı.", 
                "url": signed_url_response.get("signedUrl")
            }

    except Exception as e:
        logger.error(f"Dışa aktarma işlemi sırasında hata (export_id: {export_id}): {e}", exc_info=True)
        # Hata durumunda export kaydını güncelle
        supabase.table("project_exports").update({
            "status": "failed",
            "error_message": str(e)
        }).eq("id", export_id).execute()
        return {"success": False, "message": f"Dışa aktarma başarısız oldu: {e}"} 