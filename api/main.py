import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from fastapi import FastAPI, Depends, Request
from RequestSchemes.ScriptRequest import ScriptRequest
from RequestSchemes.VoiceoverRequest import VoiceoverRequest
from ResponseSchemes.ScriptResponse import ScriptResponse
from ResponseSchemes.VoiceoverResponse import VoiceoverResponse
from ResponseSchemes.CaptionResponse import CaptionResponse
from RequestSchemes.CaptionRequest import CaptionRequest
import write_script
import voice_over
import captions
import datetime
import platform
import psutil
import os
from auth.supabase_auth import get_current_user
from security.SanitizerMiddleware import SanitizerMiddleware
from supabase import create_client
from dotenv import load_dotenv
import logging

# Log seviyesini ayarla
logging.basicConfig(level=logging.DEBUG)
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.DEBUG)

load_dotenv()

app = FastAPI()
app.add_middleware(SanitizerMiddleware)

# API başlangıç zamanını kaydet
START_TIME = datetime.datetime.now()

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

@app.post("/script", response_model=ScriptResponse)
def generate_script(request: ScriptRequest, current_user: dict = Depends(get_current_user)):
    try:
        # Token'dan gelen user_id'yi kullan
        user_id = current_user["user_id"]
        
        script = write_script.main(
            user_id=user_id, 
            title=request.topic,
            context=request.context, 
            channel_number=request.channel_number,
        )
        
        if script is None:
            return ScriptResponse(success=False, message="Script generation failed")
        return ScriptResponse(success=True, message="Script generated successfully", script=script)
    except Exception as e:
        import traceback
        print(f"Script generation error: {str(e)}")
        traceback.print_exc()
        return ScriptResponse(success=False, message=str(e))

@app.get("/scripts", response_model=ScriptResponse)
def get_user_scripts(current_user: dict = Depends(get_current_user)):
    try:
        # Token'dan gelen user_id'yi kullan
        user_id = current_user["user_id"]
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        result = supabase.table("scripts").select("*").eq("user_id", user_id).execute()
        print(result.data)
        return ScriptResponse(success=True, message="Scripts fetched successfully", scripts=result.data)
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))

@app.get("/script/{title}", response_model=ScriptResponse)
def get_user_script_by_topic(title: str, request: Request, current_user: dict = Depends(get_current_user)):
    try:
        # Token'dan gelen user_id'yi kullan
        user_id = current_user["user_id"]
        
        # State içeriğini kontrol et
        print("Request state items:", dir(request.state))
        print("Request path params:", request.path_params)
        
        # Sanitize edilmiş title parametresini kullan
        sanitized_title = title
        
        # State'te sanitized_path_params var mı diye kontrol et
        if hasattr(request.state, "sanitized_path_params"):
            print("sanitized_path_params state'te bulundu")
            sanitized_title = request.state.sanitized_path_params.get("title", title)
            print("sanitized_title", sanitized_title)
        else:
            print("sanitized_path_params state'te bulunamadı")
            # Manuel olarak sanitize et
            try:
                from security.sanitizer import sanitize_input
                sanitized_title = sanitize_input(title, context="general")
                print("Manuel sanitize edildi:", sanitized_title)
            except Exception as e:
                print(f"Manuel sanitize hatası: {str(e)}")
        
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        result = supabase.table("scripts").select("*").eq("user_id", user_id).eq("title", sanitized_title).execute()
        if not result.data :
            return ScriptResponse(success=False, message="Script not found")
        return ScriptResponse(success=True, message="Script fetched successfully", scripts=result.data)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return ScriptResponse(success=False, message=str(e))

@app.post("/voice-over", response_model=VoiceoverResponse)
def generate_voice_over(request: VoiceoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        # Script'i güvenli bir şekilde ele al
        script_id = request.script_id
        try:
            # Sanitizer middleware ile işlenmemiş olması durumunda manuel olarak sanitize et
            from security.sanitizer import sanitize_input
            script_id = sanitize_input(script_id, context="script")
        except Exception as e:
            print(f"Script sanitize hatası: {str(e)}")
        
        channel_number = request.channel_number
        voice_over.main(user_id, script_id, channel_number)
        return VoiceoverResponse(success=True, message="Voice over generated successfully")
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))
        
@app.get("/voice-over/{id}", response_model=VoiceoverResponse)
def get_voice_over(id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        supabase_url = os.environ.get("SUPABASE_URL")
        supabase_key = os.environ.get("SUPABASE_KEY")
        supabase = create_client(supabase_url, supabase_key)
        result = supabase.table("voice_over").select("*").eq("user_id", user_id).eq("id", id).execute()
        if not result.data :
            return VoiceoverResponse(success=False, message="Voice over not found")
        return VoiceoverResponse(success=True, message="Voice over fetched successfully", voiceover=result.data[0]["voice"])
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e))

@app.post("/captions", response_model=CaptionResponse)
def generate_captions(request: CaptionRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        voice_over_id = request.voice_over_id
        captions.main(voice_over_id, user_id, request.channel_number)
        return CaptionResponse(success=True, message="Captions generated successfully")
    except Exception as e:
        return CaptionResponse(success=False, message=str(e))
