import sys
import os

from fastapi import FastAPI, Depends, Request
from api.RequestSchemes.ScriptRequest import ScriptRequest 
from api.RequestSchemes.VoiceoverRequest import VoiceoverRequest # Güncellendi
from api.ResponseSchemes.ScriptResponse import ScriptResponse, Script # Güncellendi
from api.ResponseSchemes.VoiceoverResponse import VoiceoverResponse # Güncellendi
from api.ResponseSchemes.CaptionResponse import CaptionResponse # Güncellendi
from api.RequestSchemes.CaptionRequest import CaptionRequest # Güncellendi
from api.RequestSchemes.VideoEditRequest import VideoEditRequest # Güncellendi
from api.ResponseSchemes.VideoEditResponse import VideoEditResponse # Güncellendi
from api.RequestSchemes.ProjectRequest import ProjectRequest # Güncellendi
from api.ResponseSchemes.ProjectResponse import ProjectResponse # Güncellendi
import write_script
import voice_over
import captions
import video_edit
import datetime
import platform
import psutil
import os
from api.auth.supabase_auth import get_current_user # Güncellendi
from api.security.SanitizerMiddleware import SanitizerMiddleware # Güncellendi
from supabase import create_client
from dotenv import load_dotenv
import logging
from fastapi.middleware.cors import CORSMiddleware # Eklendi

# Log seviyesini ayarla
logging.basicConfig(level=logging.DEBUG)
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.DEBUG)

load_dotenv()

app = FastAPI()

# CORS Ayarları
origins = [
    os.environ.get("FRONTEND_URL", "http://localhost:3000"), # Geliştirme URL'i eklendi
    os.environ.get("FRONTEND_PROD_URL") # Production URL
]

# ÖNEMLİ: CORSMiddleware'i *ayrı* olarak ekle
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin for origin in origins if origin], # None değerleri filtrele
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# SanitizerMiddleware'i CORS parametreleri OLMADAN ekle
app.add_middleware(SanitizerMiddleware)

# API başlangıç zamanını kaydet
START_TIME = datetime.datetime.now()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
supabase = create_client(supabase_url, supabase_key)

@app.get("/")
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

@app.post("/project", response_model=ProjectResponse)
def create_project(request: ProjectRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_name = request.project_name
        result = supabase.table("projects").insert({
            "user_id": user_id,
            "name": project_name
        }).execute()
        return ProjectResponse(success=True, message="Project created successfully", project_id=result.data[0]["id"], project_name=project_name)
    except Exception as e:
        return ProjectResponse(success=False, message=str(e))

@app.post("/script", response_model=ScriptResponse)
def generate_script(request: ScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Token'dan gelen user_id'yi kullan
        project_id = request.project_id
        
        script = write_script.main(
            project_id=project_id, 
            title=request.topic,
            context=request.context, 
            channel_number=request.channel_number,
        )
        script_data = supabase.table("projects").select("script_id").eq("id", project_id).execute()
        script_instance = Script(
            id=script_data.data[0]["script_id"],
            title=script.title,
            topic=script.topic,
            script=script.script
        )
        print(script)
        if script_instance is None:
            return ScriptResponse(success=False, message="Script generation failed")
        return ScriptResponse(success=True, message="Script generated successfully", script=script_instance)
    except Exception as e:
        import traceback
        print(f"Script generation error: {str(e)}")
        traceback.print_exc()
        return ScriptResponse(success=False, message=str(e))
@app.patch("/update-script")
def update_script(request: ScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        
        project_id = request.project_id
        result = supabase.table("projects").select("script_id").eq("id", project_id).execute()

        if result.data is None:
            return ScriptResponse(success=False, message="Project not found")
        
        script_id = result.data[0]["script_id"]
        supabase.table("scripts").update({"script": request.script}).eq("id", script_id).execute()
        return ScriptResponse(success=True, message="Script updated successfully")
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.get("/script/{project_id}", response_model=ScriptResponse)
def get_user_scripts(project_id: int, current_user: dict = Depends(get_current_user)):
    try:
        
        project_result = supabase.table("projects").select("script_id").eq("id", project_id).execute()
        if project_result.data is None:
            return ScriptResponse(success=False, message="Project not found")
        script_id = project_result.data[0]["script_id"]
        result = supabase.table("scripts").select("id, title, topic, script").eq("id", script_id).execute()
        print(result.data)
        return ScriptResponse(success=True, message="Scripts fetched successfully", scripts=result.data)
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))



@app.post("/voice-over", response_model=VoiceoverResponse)
def generate_voice_over(request: VoiceoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        # Script'i güvenli bir şekilde ele al
        project_id = request.project_id
        try:
            # Sanitizer middleware ile işlenmemiş olması durumunda manuel olarak sanitize et
            from api.security.sanitizer import sanitize_input
            project_id = sanitize_input(project_id, context="script")
        except Exception as e:
            print(f"Script sanitize hatası: {str(e)}")
        
        channel_number = request.channel_number
        voice_over_url = voice_over.main(project_id, channel_number)
        
        return VoiceoverResponse(success=True, message="Voice over generated successfully", voice_over_url=voice_over_url)
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))
        
@app.get("/voice-over/{id}", response_model=VoiceoverResponse)
def get_voice_over(id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        result = supabase.table("voice_over").select("*").eq("id", id).execute()
        if not result.data :
            return VoiceoverResponse(success=False, message="Voice over not found")
        return VoiceoverResponse(success=True, message="Voice over fetched successfully", voiceover=result.data[0]["voice"])
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))

@app.post("/caption", response_model=CaptionResponse)
def generate_captions(request: CaptionRequest, current_user: dict = Depends(get_current_user)):
    try:
        project_id = request.project_id
        captions.main(project_id, request.channel_number)
        return CaptionResponse(success=True, message="Captions generated successfully")
    except Exception as e:
        return CaptionResponse(success=False, message=str(e))

@app.post("/video-edit", response_model=VideoEditResponse)
def edit_video(request: VideoEditRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        # İstekten script_id'yi alın (VideoEditRequest şemasında olması varsayılıyor)
        project_id = request.project_id
        timeline_mode = request.use_timeline
        
        # video_edit.main'i doğru parametrelerle çağırın (user_id eklendi)
        success = video_edit.main(project_id, timeline_mode)
        
        if success:
            # Başarılı yanıt, isteğe bağlı olarak video URL'sini de içerebilir
            # (Ancak URL'yi almak için ek bir DB sorgusu gerekebilir, şimdilik basit tutuyoruz)
            return VideoEditResponse(success=True, message="Video edited and uploaded successfully")
        else:
            # Hata mesajı video_edit.main içindeki loglardan daha detaylı anlaşılabilir
            return VideoEditResponse(success=False, message="Video editing or upload process failed")
            
    except AttributeError:
        # Eğer request.script_id mevcut değilse bu hata alınabilir
        return VideoEditResponse(success=False, message="Missing 'script_id' in request body.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Hata detaylarını loglamak iyi bir pratik olabilir
        print(f"Video editing endpoint error: {str(e)}\n{error_details}")
        return VideoEditResponse(success=False, message=f"An error occurred during the video editing request: {str(e)}")
