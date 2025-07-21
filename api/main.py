import os
import tempfile

# Railway/sunucu ortamları için GCP kimlik bilgilerini ortam değişkeninden ayarla
# Bu kodun, GCP kütüphanelerini import eden diğer tüm importlardan ÖNCE çalışması gerekir.
gcp_creds_json = os.getenv("GCP_WIF_CONFIG_JSON") # Değişken adını OIDC için daha anlamlı hale getirdik
if gcp_creds_json:
    # Geçici bir dosya oluştur ve JSON içeriğini yaz
    # Dosyanın sunucu çalıştığı sürece kalması için 'delete=False' kullanıyoruz.
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as temp_creds_file:
        temp_creds_file.write(gcp_creds_json)
        creds_file_path = temp_creds_file.name
    
    # Google Cloud kütüphanelerinin kullanacağı ortam değişkenini ayarla
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = creds_file_path
    print(f"GCP Workload Identity Federation yapılandırması ortam değişkeninden okundu: {creds_file_path}")


from dotenv import load_dotenv
load_dotenv()

import json
import base64
from fastapi import FastAPI, Depends, WebSocket, WebSocketDisconnect
from urllib.parse import quote
from api.RequestSchemes.ScriptRequest import ScriptRequest 
from api.RequestSchemes.UpdateScriptRequest import UpdateScriptRequest
from api.RequestSchemes.SaveScriptRequest import SaveScriptRequest
from api.RequestSchemes.VoiceoverRequest import VoiceoverRequest 
from api.ResponseSchemes.ScriptResponse import ScriptResponse, Script 
from api.ResponseSchemes.VoiceoverResponse import VoiceoverResponse 
from api.ResponseSchemes.CaptionResponse import CaptionResponse, CaptionListResponse
from api.RequestSchemes.CaptionRequest import CaptionRequest, UpdateCaptionSegmentsRequest
from api.RequestSchemes.VideoEditRequest import VideoEditRequest 
from api.ResponseSchemes.VideoEditResponse import VideoEditResponse 
from api.RequestSchemes.ProjectRequest import ProjectRequest 
from api.ResponseSchemes.ProjectResponse import ProjectResponse 
from api.ResponseSchemes.CreateProjectRespone import CreateProject
from api.ResponseSchemes.StoryboardResponse import StoryboardResponse 
from api.RequestSchemes.StoryboardUpdateRequest import StoryboardUpdateRequest 
from api.ResponseSchemes.RefreshSignedUrl import RefreshSignedUrlResponse 
from api.RequestSchemes.UploadImageRequest import UploadImageRequest 
from api.RequestSchemes.GenerateSingleVideoRequest import GenerateSingleVideoRequest
from api.ResponseSchemes.UploadImageResponse import UploadImageResponse , ImageResponse 
from api.ResponseSchemes.GenerateSingleVideoResponse import GenerateSingleVideoResponse 
from api.RequestSchemes.GenerateTimelineVideoRequest import GenerateTimelineVideoRequest
from api.ResponseSchemes.GenerateTimelineVideoResponse import GenerateTimelineVideoResponse
from api.RequestSchemes.CreateStoryboardRequest import CreateStoryboardRequest
from api.ResponseSchemes.VideoListResponse import VideoListResponse
from api.ResponseSchemes.CreateProjectRespone import CreateProjectResponse
from api.ResponseSchemes.VoiceoverResponse import VoiceoverHistory
from api.RequestSchemes.UploadVoiceoverRequest import UploadVoiceoverRequest
from api.RequestSchemes.GenerateFinalVideoFromStoryboardRequest import GenerateFinalVideoFromStoryboardRequest
from api.ResponseSchemes.GenerateFinalVideoFromStoryboardResponse import GenerateFinalVideoFromStoryboardResponse
from api.RequestSchemes.GenerateFinalVideoRequest import GenerateFinalVideoRequest
from api.ResponseSchemes.GenerateFinalVideoResponse import GenerateFinalVideoResponse
from api.utils.generate_klingAI_video import generate_klingAI_video
from api.utils.generate_runwayML_video import generate_runwayML_video
from api.utils.generate_veo3_video import generate_veo3_video
from api.utils.generate_shots_image import generate_images_for_prompts_and_upload_to_supabase
from api.utils.upload_voiceover import upload_voiceover_to_storage
from api.utils.transcribe import transcribe_audio_bytes
from api.ResponseSchemes.ShotResponse import ShotResponse
from api.RequestSchemes.ShotRequest import ShotRequest
from api.RequestSchemes.ExportProjectRequest import ExportProjectRequest
from api.ResponseSchemes.ExportProjectResponse import ExportProjectResponse
from api.utils.export_project import export_project_assets
import write_script
from write_script import extract_topic_from_script
import voice_over
import captions
import video_edit
import datetime
import platform
import psutil
import os
from api.auth.supabase_auth import get_current_user 
from api.security.SanitizerMiddleware import SanitizerMiddleware 
from supabase import create_client
import logging
from fastapi.middleware.cors import CORSMiddleware # Eklendi
from PIL import Image # Eklendi
import io # Eklendi
from typing import Optional # Optional importu eklendi/kontrol edildi
import asyncio
import uuid
# Celery görevimizi import ediyoruz
from tasks import create_final_video_from_storyboard_task, create_final_video_without_storyboard_task
from celery.result import AsyncResult
from api.websockets.manager import manager
from api.websockets.pubsub import subscribe_to_channel

# Log seviyesini ayarla
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.DEBUG)

app = FastAPI()

# CORS Ayarları
origins = ["*"]

# ÖNEMLİ: CORSMiddleware'i *ayrı* olarak ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin], # None değerleri filtrele
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SanitizerMiddleware'i CORS parametreleri OLMADAN ekle
# app.add_middleware(SanitizerMiddleware)

# API başlangıç zamanını kaydet
START_TIME = datetime.datetime.now()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

@app.get("/health-check")
def health_check():
    # Şu anki zaman
    current_time = datetime.datetime.now()
    
    # Çalışma süresi hesaplaması
    uptime = current_time - START_TIME
    uptime_seconds = uptime.total_seconds()
    uptime_str = str(datetime.timedelta(seconds=int(uptime_seconds)))
    
    # Sistem bilgileri
    system_info = {
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "processor": platform.processor()
    }
    
    # Bellek kullanımı
    memory = psutil.virtual_memory()
    memory_info = {
        "total": f"{memory.total / (1024**3):.2f} GB",
        "available": f"{memory.available / (1024**3):.2f} GB",
        "used_percent": f"{memory.percent}%"
    }
    
    # CPU kullanımı
    cpu_info = {
        "cpu_percent": f"{psutil.cpu_percent()}%",
        "cpu_count": psutil.cpu_count()
    }
    
    # Disk kullanımı
    disk = psutil.disk_usage('/')
    disk_info = {
        "total": f"{disk.total / (1024**3):.2f} GB",
        "free": f"{disk.free / (1024**3):.2f} GB",
        "used_percent": f"{disk.percent}%"
    }
    
    return {
        "status": "up",
        "service": "VideoAI API",
        "message": "Sistem çalışıyor",
        "version": "1.0.0",
        "timestamp": current_time.isoformat(),
        "uptime": uptime_str,
        "system": system_info,
        "memory": memory_info,
        "cpu": cpu_info,
        "disk": disk_info,
        "environment": os.environ.get("ENVIRONMENT", "development")
    }



@app.post("/project", response_model=CreateProjectResponse)
def create_project(request: ProjectRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_name = request.project_name
        result = supabase.table("projects").insert({
            "user_id": user_id,
            "name": project_name
        }).execute()
        print(result.data[0])
        return CreateProjectResponse(success=True, message="Project created successfully", project=CreateProject(id=result.data[0]["id"], name=project_name, user_id=user_id, created_at=result.data[0]["created_at"]))
    except Exception as e:
        return CreateProjectResponse(success=False, message=str(e))
    

@app.get("/projects", response_model=ProjectResponse)
def get_projects(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        result = supabase.table("projects").select("id, name").eq("user_id", user_id).execute()
        return ProjectResponse(success=True, message="Projects fetched successfully", projects=result.data)
    except Exception as e:
        return ProjectResponse(success=False, message=str(e))


@app.post("/save-script", response_model=ScriptResponse)
def save_script(request: SaveScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        project_id = request.project_id
        script = request.script
        user_id = current_user["user_id"]
        topic = extract_topic_from_script(script).topic
        result = supabase.table("scripts").insert({"project_id": project_id, "script": script, "topic": topic,"user_id":user_id}).execute()
        print(result)
        if result.data is None:
            return ScriptResponse(success=False, message="Script saved failed")
        return ScriptResponse(success=True, message="Script saved successfully", script_id=result.data[0]["id"])
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))
    

@app.post("/script", response_model=ScriptResponse)
def generate_script(request: ScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Token'dan gelen user_id'yi kullan
        user_id = current_user["user_id"]
        project_id = request.project_id
        script = write_script.main(
            user_id=user_id,
            project_id=project_id, 
            title=request.topic,
            context=request.context, 
            channel_number=request.channel_number,
            tone=request.tone
        )
        
       
        print(script)
        if script is None:
            return ScriptResponse(success=False, message="Script generation failed")
        return ScriptResponse(success=True, message="Script generated successfully", script=script)
    except Exception as e:
        import traceback
        print(f"Script generation error: {str(e)}")
        traceback.print_exc()
        return ScriptResponse(success=False, message=str(e))
    

@app.put("/update-script",response_model=ScriptResponse)
def update_script(request: UpdateScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        print(request)
        script_id = request.script_id
        supabase.table("scripts").update({"script": request.script}).eq("id", script_id).execute()
        return ScriptResponse(success=True, message="Script updated successfully")
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.get("/script/{project_id}", response_model=ScriptResponse)
def get_user_scripts(project_id: int, current_user: dict = Depends(get_current_user)):
    try:
        result = supabase.table("scripts").select("id, title, topic, script, created_at").eq("user_id", current_user["user_id"]).filter("project_id", "eq", project_id).execute()
        return ScriptResponse(success=True, message="Scripts fetched successfully", scripts=result.data)
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.get("/script/{script_id}", response_model=ScriptResponse)
def get_script(script_id: int, current_user: dict = Depends(get_current_user)):
    try:
        result = supabase.table("scripts").select("id, title, topic, script, created_at").eq("id", script_id).execute()
        return ScriptResponse(success=True, message="Script fetched successfully", script=result.data[0])
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.get("/script-properties/{project_id}",response_model=ScriptResponse)
def get_script_properties(project_id: int, current_user: dict = Depends(get_current_user)):
    try:
        result = supabase.table("scripts").select("id, topic, script, created_at").eq("project_id", project_id).execute()
        return ScriptResponse(success=True, message="Script property fetched successfully", scripts=result.data)
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.post("/voice-over", response_model=VoiceoverResponse)
def generate_voice_over(request: VoiceoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        script_id = request.script_id
        similarity_boost = request.similarity_boost
        stability = request.stability
        project_id = request.project_id
        voice_id = request.voice_id
        voice_over_url,voice_over_id = voice_over.main(user_id,project_id,script_id,similarity_boost, stability,voice_id)
        logging.info(f"Voice over URL: {voice_over_url}")
        print(f"Voice over URL: {voice_over_url}")
        return VoiceoverResponse(success=True, message="Voice over generated successfully", voice_over_url=voice_over_url,voice_over_id=voice_over_id)
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e)), 500
    

@app.post("/upload-voice-over", response_model=VoiceoverResponse)
def upload_voice_over(request: UploadVoiceoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_id = request.project_id
        audio_file_base64 = request.audio_file
        voice_over_name = request.voice_over_name
        
        voice_over_data = upload_voiceover_to_storage(user_id, project_id, audio_file_base64, voice_over_name)
        
        if voice_over_data is None: 
            return VoiceoverResponse(success=False, message="Voice over upload failed")
        
        # Transcribe the audio
        audio_bytes = base64.b64decode(audio_file_base64)
        transcribed_text = transcribe_audio_bytes(audio_bytes)

        if transcribed_text:
            topic = extract_topic_from_script(transcribed_text).topic
            script_insert_result = supabase.table("scripts").insert({
                "project_id": project_id,
                "script": transcribed_text,
                "topic": topic,
                "user_id": user_id,
            }).execute()

            if not script_insert_result.data:
               
                return VoiceoverResponse(success=False, message="Failed to save transcribed script.")

            # voice_over tablosunu yeni script_id ile güncelle
            script_id = script_insert_result.data[0]["id"]
            voiceover_data_update = supabase.table("voice_over").update({
                "script_id": script_id
            }).eq("id", voice_over_data["id"]).execute()

            if not voiceover_data_update.data:
                
                return VoiceoverResponse(success=False, message="Failed to link voice over to the new script.")

        voice_over_url = supabase.storage.from_("voice-over-files").create_signed_url(voice_over_name,3600)
        return VoiceoverResponse(success=True, message="Voice over uploaded and transcribed successfully", voice_over_history=[VoiceoverHistory(id=voice_over_data["id"], name=voice_over_data["voice_name"], duration=voice_over_data["duration"], url=voice_over_url.get("signedURL")  , created_at=voice_over_data["created_at"])])
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e)), 500


@app.get("/voice-over/history/{project_id}", response_model=VoiceoverResponse)
def get_voice_over_history(project_id: int, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        voice_urls = []
        result = supabase.table("voice_over").select("id,voice_name, duration, created_at").eq("user_id", user_id).eq("project_id", project_id).execute()
        print(f"Voice over history: {result.data}")
        for vname in result.data:
            voice_over_url = supabase.storage.from_("voice-over-files").create_signed_url(vname["voice_name"],3600)
            voice_urls.append(VoiceoverHistory(id=vname["id"], name=vname["voice_name"], duration=vname["duration"], url=voice_over_url.get("signedURL"), created_at=vname["created_at"]))
        print(f"Voice over history: {voice_urls}")
        return VoiceoverResponse(success=True, message="Voice over history fetched successfully", voice_over_history=voice_urls)
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))
    

@app.get("/voice-over/{id}", response_model=VoiceoverResponse)
def get_voice_over(id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        result = supabase.table("voice_over").select("*").eq("id", id).execute()
        if not result.data :
            return VoiceoverResponse(success=False, message="Voice over not found")
        voice_over_url = supabase.storage.from_("voice-over-files").create_signed_url(result.data[0]["voice_name"],3600)
        return VoiceoverResponse(success=True, message="Voice over fetched successfully", voice_over_url=voice_over_url.get("signedURL"))
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))


@app.post("/caption", response_model=CaptionResponse)
def generate_captions(request: CaptionRequest, current_user: dict = Depends(get_current_user)):
    try:
        project_id = request.project_id
        voice_over_id = request.voice_over_id
        user_id = current_user["user_id"]
        print(f"Main User id: {user_id}")
        caption_id = captions.main(project_id, voice_over_id, request.channel_number,user_id=user_id)
        if caption_id is None:
            return CaptionResponse(success=False, message="Captions generation failed")
        
        # Caption verilerini formatla ve döndür
        formatted_caption = captions.format_caption_for_api(supabase, caption_id)
        if formatted_caption:
            return CaptionResponse(
                success=True, 
                message="Captions generated successfully", 
                id=caption_id,
                caption=formatted_caption
            )
        else:
            return CaptionResponse(success=True, message="Captions generated successfully", id=caption_id)
    except Exception as e:
        import traceback
        logger.error(f"Error during caption generation: {e}")
        logger.error(traceback.format_exc())
        return CaptionResponse(success=False, message=f"An unexpected error occurred: {e}")


@app.get("/caption/{caption_id}", response_model=CaptionResponse)
def get_caption_details(caption_id: int, current_user: dict = Depends(get_current_user)):
    """
    Belirli bir caption'ın detaylarını getirir
    """
    try:
        formatted_caption = captions.format_caption_for_api(supabase, caption_id)
        if formatted_caption:
            return CaptionResponse(
                success=True,
                message="Caption details fetched successfully",
                id=caption_id,
                caption=formatted_caption
            )
        else:
            return CaptionResponse(success=False, message="Caption not found")
    except Exception as e:
        return CaptionResponse(success=False, message=str(e))


@app.get("/captions/project/{project_id}", response_model=CaptionListResponse)
def get_project_captions(project_id: int, current_user: dict = Depends(get_current_user)):
    """
    Belirli bir projeye ait tüm caption'ları getirir
    """
    try:
        captions_list = captions.get_captions_by_project_for_api(supabase, project_id)
        return CaptionListResponse(
            success=True,
            message="Project captions fetched successfully",
            captions=captions_list
        )
    except Exception as e:
        return CaptionListResponse(success=False, message=str(e))


@app.put("/caption/{caption_id}/segments", response_model=CaptionResponse)
def update_caption_segments(
    caption_id: int, 
    request: UpdateCaptionSegmentsRequest, 
    current_user: dict = Depends(get_current_user)
):
    """
    Caption segment'lerini günceller
    """
    try:
        # Segments'leri dict formatına çevir
        segments_data = []
        for segment in request.segments:
            words_data = []
            for word in segment.words:
                words_data.append({
                    "word": word.word,
                    "start": word.start,
                    "end": word.end
                })
            
            segments_data.append({
                "id": segment.id,
                "text": segment.text,
                "start": segment.start,
                "end": segment.end,
                "words": words_data
            })
        
        # Veritabanını güncelle
        success = captions.update_caption_segments_in_db(supabase, caption_id, segments_data)
        
        if success:
            # Güncellenmiş veriyi getir ve döndür
            formatted_caption = captions.format_caption_for_api(supabase, caption_id)
            return CaptionResponse(
                success=True,
                message="Caption segments updated successfully",
                id=caption_id,
                caption=formatted_caption
            )
        else:
            return CaptionResponse(success=False, message="Failed to update caption segments")
            
    except Exception as e:
        return CaptionResponse(success=False, message=str(e))



            
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"Video production endpoint error: {str(e)}\n{error_details}")
        return VideoEditResponse(success=False, message=f"An error occurred during the video production request: {str(e)}")


@app.post("/create-storyboard", response_model=StoryboardResponse)
def create_storyboard(request: CreateStoryboardRequest, current_user: dict = Depends(get_current_user)):
    try:
        shots = []
        prompts = []
        storyboard_name = request.name
        project_id = request.project_id
        caption_id = request.caption_id
        user_id = current_user["user_id"]
        logger.info(f"Creating storyboard for project_id {project_id} with caption_id {caption_id} and user_id {user_id}")
        storyboard_result = supabase.table("storyboards").insert({
            "name":storyboard_name,
            "user_id":user_id,
            "project_id":project_id,
        }).execute()
        logger.info(f"Storyboard created with id {storyboard_result.data[0]['id']}")
        storyboard_id = storyboard_result.data[0]["id"]
        success = video_edit.prepare_video_assets(project_id, caption_id, user_id)
        logger.info(f"Video assets prepared: {success}")
        if not success:
            return StoryboardResponse(success=False, message="Failed to prepare video assets.")
        
        response_json_result = supabase.table("projects").select("response_json").eq("id", project_id).execute()

        # response_json varlığını ve içeriğini kontrol et
        if response_json_result.data and response_json_result.data[0].get("response_json"):
            response_json = response_json_result.data[0]["response_json"]
            try:
                response_json_data = json.loads(response_json)
                if not isinstance(response_json_data, list): # Beklenen format liste değilse
                    print(f"Warning: response_json for project_id {project_id} is not a list. Proceeding with empty shots.")
                    response_json_data = [] # Boş liste ile devam et
            except json.JSONDecodeError:
                print(f"Warning: Invalid JSON in response_json for project_id {project_id}. Proceeding with empty shots.")
                response_json_data = [] # JSON parse hatası durumunda boş liste

            for index, clip_data in enumerate(response_json_data):
               # clip_data'nın beklenen anahtarları içerip içermediğini kontrol et
               if not all(key in clip_data for key in ["clip_name", "suggestion", "duration", "explanation", "script_segment", "start_time"]):
                   print(f"Warning: Missing keys in clip_data for project_id {project_id}, storyboard_id {storyboard_id}. Skipping this shot.")
                   continue

               signed_url_data = supabase.storage.from_("video-database").create_signed_url(clip_data["clip_name"],3600)
               print(signed_url_data)
               video_url = signed_url_data.get('signedURL') if signed_url_data else None
               prompts.append(clip_data["suggestion"])
               shot_insert_result = supabase.table("shots").insert({
                   "approved":False,
                   "video_url": video_url,
                   "duration":clip_data["duration"],
                   "suggestion":clip_data["suggestion"],
                   "explanation":clip_data["explanation"],
                   "clip_name":clip_data["clip_name"],
                   "script_segment":clip_data["script_segment"],
                   "start_time":clip_data["start_time"],
                   "shot_index":index,
                   "storyboard_id":storyboard_id,
                   "user_id": current_user["user_id"]
               }).execute()
               # shot_insert_result'ın başarılı olup olmadığını kontrol et
               if shot_insert_result.data:
                   shots.append(shot_insert_result.data[0])
               else:
                   print(f"Warning: Failed to insert shot for project_id {project_id}, storyboard_id {storyboard_id}. Error: {shot_insert_result.error}")
            
            # Her bir çekim için paralel olarak birer görsel oluştur.
            # Bu işlem endpoint'in yanıtını bekletmez.
            if prompts:
                print(f"Storyboard {storyboard_id} için {len(prompts)} adet görsel oluşturma işlemi başlatılıyor.")
                # Görsel oluşturma işlemini başlat, ancak sonucunu bekleme (fire-and-forget)
                try:
                    # Not: Bu fonksiyon içindeki ThreadPoolExecutor sayesinde paralel çalışır.
                    generate_images_for_prompts_and_upload_to_supabase(
                        prompts=prompts, 
                        user_id=current_user["user_id"], 
                        storyboard_id=storyboard_id,
                        project_id=project_id
                    )
                    print(f"Storyboard {storyboard_id} için görsel oluşturma görevleri başarıyla gönderildi.")
                except Exception as img_exc:
                    # Bu hatayı logla ama endpoint'in başarısız olmasına neden olma,
                    # çünkü storyboard ve shot'lar başarıyla oluşturuldu.
                    print(f"Error initiating image generation for storyboard {storyboard_id}: {img_exc}")

        else:
            # response_json yoksa veya null ise, shot_index_size parametresini kontrol et
            print(f"No valid response_json found for project_id {project_id}. Checking shot_index_size.")
            if request.shot_index_size is not None and request.shot_index_size > 0:
                print(f"Creating {request.shot_index_size} empty shots based on shot_index_size.")
                for i in range(request.shot_index_size):
                    shot_insert_result = supabase.table("shots").insert({
                        "approved": False,
                        "video_url": None,
                        "duration": None,
                        "suggestion": "Empty shot", # Veya boş bırakılabilir
                        "explanation": "Automatically generated empty shot", # Veya boş bırakılabilir
                        "clip_name": None,
                        "script_segment": None,
                        "start_time": None,
                        "shot_index": i,
                        "storyboard_id": storyboard_id,
                        "user_id": current_user["user_id"]
                        # Diğer gerekli alanlar varsa None veya varsayılan değerlerle eklenebilir
                    }).execute()
                    if shot_insert_result.data:
                        shots.append(shot_insert_result.data[0])
                    else:
                        print(f"Warning: Failed to insert empty shot index {i} for storyboard_id {storyboard_id}. Error: {shot_insert_result.error}")
            else:
                print(f"shot_index_size is not provided or invalid. No empty shots will be created.")
                # prompts ve shots boş kalacak, generate_images_for_prompts_and_upload_to_supabase çağrılmayacak

        return StoryboardResponse(success=True, message="Storyboard created successfully", storyboards=[{"id":storyboard_id,"project_id":project_id, "name": storyboard_name, "shots":shots}])
    except Exception as e:
        import traceback # Detaylı hata takibi için
        print(f"Error in create_storyboard: {str(e)}\n{traceback.format_exc()}") # Hata loglaması
        return StoryboardResponse(success=False, message=str(e), storyboards=[])

@app.get("/user-storyboards/{project_id}", response_model=StoryboardResponse)
def get_storyboards(project_id: int, current_user: dict = Depends(get_current_user)):
    try:
        print(project_id)
        user_id = current_user["user_id"]
        result = supabase.table("storyboards").select("id, project_id, name, created_at, updated_at").eq("user_id", user_id).eq("project_id", project_id).execute()
        print(result.data)
        return StoryboardResponse(success=True, message="Storyboards fetched successfully", storyboards=result.data)
    except Exception as e:
        return StoryboardResponse(success=False, message=str(e), storyboards=[])


@app.get("/storyboards/{storyboard_id}", response_model=StoryboardResponse)
def get_storyboard(storyboard_id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        storyboard_id_int = int(storyboard_id)
        
        result = supabase.table("storyboards").select("id, project_id, name, created_at, updated_at").eq("user_id", user_id).eq("id", storyboard_id_int).single().execute()
        
        if not result.data:
            return StoryboardResponse(success=False, message="Storyboard not found", storyboards=[])

        # Storyboard bulunduktan sonra çekimleri al
        shot_result = supabase.table("shots").select("*").eq("storyboard_id",storyboard_id_int).execute()
        
        # Çekim olmasa bile bu bir hata değildir, boş bir liste olabilir.
        shots = shot_result.data if shot_result.data else []

        storyboard_data = result.data
        storyboard_data["shots"] = shots
          
        return StoryboardResponse(success=True, message="Storyboard fetched successfully", storyboards=[storyboard_data])
    except ValueError:
        return StoryboardResponse(success=False, message="Invalid storyboard ID format", storyboards=[])
    except Exception as e:
        print(f"Error in get_storyboard: {str(e)}")  # Hata loglaması ekle
        return StoryboardResponse(success=False, message=str(e), storyboards=[])



@app.put("/storyboards-update/{storyboard_id}", response_model=StoryboardResponse)
def update_storyboard(storyboard_id: str, request: StoryboardUpdateRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        update_data = request.model_dump(exclude_unset=True)
        result = supabase.table("storyboards").update(update_data).eq("user_id", user_id).eq("id", storyboard_id).execute()
        return StoryboardResponse(success=True, message="Storyboard updated successfully", storyboards=[result.data[0]])
    except Exception as e:
        return StoryboardResponse(success=False, message=str(e), storyboards=[])


@app.get("/refresh-video-url/{clip_name}")
def refresh_signed_url(clip_name: str):
    try:
        signed_url_raw = supabase.storage.from_("video-database").create_signed_url(quote(clip_name), 3600)
        signed_url = signed_url_raw.get('signedUrl')
        print(signed_url)
        return RefreshSignedUrlResponse(success=True, message="Signed URL refreshed successfully", signed_url=signed_url)
    except Exception as e:
        return RefreshSignedUrlResponse(success=False, message=str(e))



@app.post("/upload-image", response_model=UploadImageResponse)
def upload_image(request: UploadImageRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        image_base64 = request.image
        print(image_base64)
        # Base64 başlığını (örn: "data:image/png;base64,") kaldır
        if "," in image_base64:
            image_base64 = image_base64.split(",")[1]
        
        image_bytes = base64.b64decode(image_base64)
        
       
        try:
            img = Image.open(io.BytesIO(image_bytes))
            width, height = img.size
            aspect_ratio = f"{height}x{width}"
        except Exception as e:
            # Resim işleme hatası durumunda varsayılan veya hata değeri ata
            print(f"Error processing image for aspect ratio: {str(e)}")
            aspect_ratio = "unknown"


        image_path = f"images/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.png" # Kullanıcıya özel klasör
        
        
        storage_client = supabase.storage.from_("videos") 
        upload_result = storage_client.upload(path=image_path, file=image_bytes, file_options={"content-type": "image/png"})
        
        # Yükleme sonucunu kontrol et
       

        # Supabase Storage'dan dosyanın genel URL'sini al
        # Not: public_url kullanımı için bucket'ınızın public olması veya RLS kurallarınızın izin vermesi gerekir.
        # create_signed_url daha güvenli bir seçenek olabilir.
        image_public_url = storage_client.get_public_url(image_path)
        print(f"Image uploaded to: {image_public_url}")
        # Geçici olarak create_signed_url kullanalım (daha güvenli)
        # signed_url_response = storage_client.create_signed_url(image_path, 3600) # 1 saat geçerli URL
        # if not signed_url_response or 'signedURL' not in signed_url_response:
        #     print(f"Failed to create signed URL for {image_path}")
        #     return UploadImageResponse(success=False, message="Failed to get image URL after upload.")
        
        # image_accessible_url = signed_url_response['signedURL']


        result = supabase.table("images").insert({
            "user_id": user_id,
            "url": image_public_url, 
            "size": aspect_ratio,
        }).execute()
        
        if not result.data:
            # Veritabanı ekleme hatası
            print(f"Failed to insert image metadata to database: {result.error}")
            return UploadImageResponse(success=False, message="Failed to save image metadata.")

        return UploadImageResponse(success=True, message="Image uploaded successfully", image_response=ImageResponse(id=result.data[0]["id"], url=image_public_url))
    except Exception as e:
        import traceback
        print(f"Error in upload_image: {str(e)}")
        traceback.print_exc()
        return UploadImageResponse(success=False, message=str(e))
    


@app.post("/generate-single-video")
async def generate_single_video(request: GenerateSingleVideoRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        model = request.model
        video = None


        match model:
            case "klingai":
                video = await generate_klingAI_video(request.prompt, request.negativePrompt, request.duration, request.cfgScale, request.aspectRatio, request.startImage, request.endImage)
            case "runwayml":
                video = await generate_runwayML_video(request.prompt, request.duration, request.aspectRatio, request.startImage, request.endImage)
            case "veo3":
                video = await generate_veo3_video(request.prompt, request.duration, request.aspectRatio, request.negativePrompt, sampleCount=1)
            case _:
                return GenerateSingleVideoResponse(success=False, message=f"Desteklenmeyen model türü: {model}")

        print(video)
        if video is None:
            return GenerateSingleVideoResponse(success=False, message="Video generation failed")
        
        video_name = f"videos/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
        video_upload_result = supabase.storage.from_("videos").upload(path=video_name, file=video)


        

        video_url = supabase.storage.from_("videos").get_public_url(video_name)
        video_insert_result = supabase.table("generated_videos").insert({
            "user_id": user_id,
            "name": video_name
        }).execute()


        video_id = video_insert_result.data[0]["id"]

        if request.startImageId is not None:
            update_start_image_result = supabase.table("uploaded_images").update({
                "video_id": video_id
            }).eq("id", request.startImageId).execute()

        if request.endImageId is not None:
            update_end_image_result = supabase.table("uploaded_images").update({
                "video_id": video_id
            }).eq("id", request.endImageId).execute()

        return GenerateSingleVideoResponse(success=True, message="Video generated successfully", video=video_url)
    except Exception as e:
        return GenerateSingleVideoResponse(success=False, message=str(e))


@app.post("/generate-timeline-video")
async def generate_timeline_video(request: GenerateTimelineVideoRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        model = request.model
        project_id = request.project_id
        batch_id = str(uuid.uuid4()) # Her istek için benzersiz bir batch_id oluştur
        
        generation_tasks = []
        for segment in request.segments:
            task = None
            match model:
                case "klingai":
                    task = generate_klingAI_video(segment.prompt, None, segment.duration, segment.cfg_scale, request.aspect_ratio, segment.start_image, segment.end_image)
                case "runwayml":
                    task = generate_runwayML_video(segment.prompt, segment.duration, request.aspect_ratio, segment.start_image, segment.end_image)
                case "veo3":
                    task = generate_veo3_video(segment.prompt, segment.duration, request.aspect_ratio, None, sampleCount=1)
                case _:
                    return GenerateTimelineVideoResponse(success=False, message=f"Desteklenmeyen model türü: {model}")
            
            if task:
                generation_tasks.append(task)
        
        if not generation_tasks:
            return GenerateTimelineVideoResponse(success=False, message=f"Desteklenmeyen veya geçersiz model türü: {model}")

        # Tüm video oluşturma görevlerini paralel olarak çalıştır
        generated_videos_content = await asyncio.gather(*generation_tasks, return_exceptions=True)
        
        videos = []
        # Not: Supabase işlemleri şimdilik senkron kalacak. FastAPI bunları bir thread pool'da çalıştıracaktır.
        for i, result in enumerate(generated_videos_content):
            segment = request.segments[i]

            if isinstance(result, Exception):
                print(f"Segment {i} için video oluşturma başarısız oldu: {result}")
                # Başarısız olan segmenti atla veya isteğe bağlı olarak bir hata nesnesi döndür
                continue

            video_content = result
            if video_content is None:
                print(f"Segment {i} için video içeriği boş geldi.")
                continue

            video_name = f"videos/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}-{segment.start_image_id}.mp4"
            
            try:
                # Supabase'e yükleme
                supabase.storage.from_("videos").upload(path=video_name, file=video_content)
                video_url = supabase.storage.from_("videos").get_public_url(video_name)
                videos.append(video_url)
                
                # Veritabanına ekleme
                supabase.table("generated_videos").insert({
                    "user_id": user_id,
                    "name": video_name,
                    "batch_id": batch_id,
                    "path": video_name,
                    "start_image_id": segment.start_image_id,
                    "end_image_id": segment.end_image_id,
                    "project_id": project_id
                }).execute()
            except Exception as e:
                print(f"Segment {i} için Supabase işlemi başarısız oldu: {e}")
                # Bu segment için hatayı logla ve devam et

        if not videos:
            return GenerateTimelineVideoResponse(success=False, message="Tüm segmentler için video oluşturma veya yükleme başarısız oldu.")

        return GenerateTimelineVideoResponse(success=True, message="Video oluşturma başarıyla tamamlandı.", video_urls=videos, segments=request.segments)
        
    except Exception as e:
        return GenerateTimelineVideoResponse(success=False, message=str(e))

@app.get("/video-list", response_model=VideoListResponse)
def get_video_list(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        # user_id'ye göre video bilgilerini seç
        result = supabase.table("generated_videos").select("id,name,created_at,batch_id").eq("user_id", user_id).order("created_at", desc=True).execute()

        if result.data is not None:
            # Eğer result.data boş bir liste değilse ve içinde öğeler varsa
            if result.data:
                for video in result.data:
                    # 'name' alanını kullanarak public URL'i oluştur ve video sözlüğüne ekle
                    video['url'] = supabase.storage.from_("videos").get_public_url(video['name'])
                return VideoListResponse(success=True, message="Video list fetched successfully", videos=result.data)
            else:
                 # Veri var ama boş liste (kullanıcının videosu yok)
                return VideoListResponse(success=True, message="No videos found for this user.", videos=[])
        elif result.error:
            print(f"Supabase error fetching video list: {result.error}")
            # Pydantic modelinin videos alanı Optional olduğu için hata durumunda videos=None veya videos=[] olabilir.
            return VideoListResponse(success=False, message=f"Failed to fetch video list: {result.error.message if hasattr(result.error, 'message') else str(result.error)}", videos=None)
        else:
            # Bu durum pek olası değil (data None ve error None), ama yine de kapsayalım
            return VideoListResponse(success=True, message="No videos found or an unknown state occurred.", videos=[])

    except Exception as e:
        import traceback
        print(f"Exception in get_video_list: {str(e)}\n{traceback.format_exc()}")
        return VideoListResponse(success=False, message=str(e), videos=None)

@app.patch("/shots/{shot_id}")
def change_shot_approved_status(shot_id: str, request: ShotRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        is_approved = request.is_approved
        result = supabase.table("shots").update({"approved": is_approved}).eq("user_id", user_id).eq("id", shot_id).execute()
        if result.data:
            return ShotResponse(success=True, message="Shot updated successfully", shot=result.data[0])
        else:
            return ShotResponse(success=False, message="Shot not found", shot=None)
    except Exception as e:
        return ShotResponse(success=False, message=str(e), shot=None)





@app.post("/generate-final-video-from-storyboard", status_code=202, response_model=GenerateFinalVideoFromStoryboardResponse)
def generate_final_video_from_storyboard(request: GenerateFinalVideoFromStoryboardRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        storyboard_id = request.storyboard_id
        
        # Storyboard varlığını ve kullanıcıya ait olduğunu kontrol et
        storyboard_response = supabase.table("storyboards").select("id, project_id").eq("id", storyboard_id).eq("user_id", user_id).single().execute()
        
        if not storyboard_response.data:
            return GenerateFinalVideoFromStoryboardResponse(success=False, message="Storyboard not found or access denied.")
            
        project_id = storyboard_response.data["project_id"]

        # Ağır video oluşturma işini doğrudan çağırmak yerine,
        # Celery görevini kuyruğa gönderiyoruz.
        logger.info(f"Queueing video generation task for storyboard {storyboard_id}...")
        task = create_final_video_from_storyboard_task.delay(
            storyboard_id=storyboard_id,
            project_id=project_id,
            user_id=user_id
        )
        logger.info(f"Task queued successfully with ID: {task.id}")

        # Kullanıcıya görevin başladığını ve görev ID'sini hemen döndürüyoruz.
        return GenerateFinalVideoFromStoryboardResponse(
            success=True, 
            message="Final video generation has been started.",
            task_id=task.id
        )

    except Exception as e:
        import traceback
        logger.error(f"Error in generate_final_video_from_storyboard endpoint: {str(e)}\n{traceback.format_exc()}")
        return GenerateFinalVideoFromStoryboardResponse(success=False, message=f"An unexpected error occurred while queueing the task: {str(e)}")

@app.post("/generate-final-video", status_code=202, response_model=GenerateFinalVideoResponse)
def generate_final_video(request: GenerateFinalVideoRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_id = request.project_id
        caption_id = request.caption_id

        # Proje ve caption'ın varlığını ve kullanıcıya ait olduğunu kontrol et (opsiyonel ama önerilir)
        project_check = supabase.table("projects").select("id").eq("id", project_id).eq("user_id", user_id).single().execute()
        if not project_check.data:
            return GenerateFinalVideoResponse(success=False, message="Project not found or access denied.")
            
        caption_check = supabase.table("captions").select("id").eq("id", caption_id).eq("project_id", project_id).single().execute()
        if not caption_check.data:
            return GenerateFinalVideoResponse(success=False, message="Caption not found or it does not belong to the specified project.")

        logger.info(f"Queueing video generation task for project {project_id} and caption {caption_id}...")
        task = create_final_video_without_storyboard_task.delay(
            project_id=project_id,
            caption_id=caption_id,
            user_id=user_id
        )
        logger.info(f"Task queued successfully with ID: {task.id}")

        return GenerateFinalVideoResponse(
            success=True,
            message="Final video generation has been started.",
            task_id=task.id
        )
    except Exception as e:
        import traceback
        logger.error(f"Error in generate_final_video endpoint: {str(e)}\n{traceback.format_exc()}")
        return GenerateFinalVideoResponse(success=False, message=f"An unexpected error occurred while queueing the task: {str(e)}")


@app.websocket("/ws/task-status/{task_id}")
async def websocket_endpoint(websocket: WebSocket, task_id: str):
    """
    WebSocket bağlantısını yönetir.
    - Bir task_id için bağlantı kurar.
    - Redis Pub/Sub üzerinden o task_id kanalını dinler.
    - Gelen mesajları istemciye iletir.
    """
    await manager.connect(task_id, websocket)
    logger.info(f"WebSocket connection established for task_id: {task_id}")
    
    try:
        # Redis'ten gelen mesajları dinle ve istemciye gönder
        async for message in subscribe_to_channel(task_id):
            logger.info(f"Message received from Redis for {task_id}: {message}")
            # Mesajın JSON olup olmadığını kontrol et
            try:
                # JSON verisi ise, JSON olarak gönder
                data = json.loads(message)
                await manager.send_json_message(data, task_id)
            except json.JSONDecodeError:
                # Düz metin ise, metin olarak gönder
                await manager.send_personal_message(message, task_id)

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task_id: {task_id}")
    except Exception as e:
        logger.error(f"An error occurred in WebSocket for task_id {task_id}: {e}", exc_info=True)
    finally:
        # Bağlantı koptuğunda veya hata olduğunda bağlantıyı temizle
        manager.disconnect(task_id)
        logger.info(f"Connection for task_id {task_id} closed and cleaned up.")





@app.post("/export-project", response_model=ExportProjectResponse)
def export_project(request: ExportProjectRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_id = request.project_id
        
        result = export_project_assets(project_id, user_id, supabase)
        
        return ExportProjectResponse(success=result["success"], message=result["message"], export_url=result.get("url"))
        
    except Exception as e:
        return ExportProjectResponse(success=False, message=str(e))
        
        


    