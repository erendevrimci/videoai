import json
import os
import base64
from fastapi import FastAPI, Depends
from urllib.parse import quote
from api.RequestSchemes.ScriptRequest import ScriptRequest 
from api.RequestSchemes.UpdateScriptRequest import UpdateScriptRequest
from api.RequestSchemes.SaveScriptRequest import SaveScriptRequest
from api.RequestSchemes.VoiceoverRequest import VoiceoverRequest 
from api.ResponseSchemes.ScriptResponse import ScriptResponse, Script 
from api.ResponseSchemes.VoiceoverResponse import VoiceoverResponse 
from api.ResponseSchemes.CaptionResponse import CaptionResponse 
from api.RequestSchemes.CaptionRequest import CaptionRequest 
from api.RequestSchemes.VideoEditRequest import VideoEditRequest 
from api.ResponseSchemes.VideoEditResponse import VideoEditResponse 
from api.RequestSchemes.ProjectRequest import ProjectRequest 
from api.ResponseSchemes.ProjectResponse import ProjectResponse 
from api.ResponseSchemes.ResponseJson import ResponseJson 
from api.RequestSchemes.StoryboardRequest import StoryboardRequest 
from api.ResponseSchemes.StoryboardResponse import StoryboardResponse 
from api.RequestSchemes.StoryboardUpdateRequest import StoryboardUpdateRequest 
from api.ResponseSchemes.RefreshSignedUrl import RefreshSignedUrlResponse 
from api.RequestSchemes.UploadImageRequest import UploadImageRequest 
from api.RequestSchemes.GenerateSingleVideoRequest import GenerateSingleVideoRequest
from api.ResponseSchemes.UploadImageResponse import UploadImageResponse , ImageResponse 
from api.ResponseSchemes.GenerateSingleVideoResponse import GenerateSingleVideoResponse 
from api.RequestSchemes.GenerateTimelineVideoRequest import GenerateTimelineVideoRequest
from api.ResponseSchemes.GenerateTimelineVideoResponse import GenerateTimelineVideoResponse
from api.ResponseSchemes.VideoListResponse import VideoListResponse
from api.utils.generate_klingAI_video import generate_klingAI_video
from api.utils.generate_runwayML_video import generate_runwayML_video

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
from dotenv import load_dotenv
import logging
from fastapi.middleware.cors import CORSMiddleware # Eklendi
from PIL import Image # Eklendi
import io # Eklendi
from typing import Optional # Optional importu eklendi/kontrol edildi

# Log seviyesini ayarla
logging.basicConfig(level=logging.DEBUG)
security_logger = logging.getLogger("security")
security_logger.setLevel(logging.DEBUG)

load_dotenv()

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



@app.post("/project", response_model=ProjectResponse)
def create_project(request: ProjectRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        project_name = request.project_name
        result = supabase.table("projects").insert({
            "user_id": user_id,
            "name": project_name
        }).execute()
        return ProjectResponse(success=True, message="Project created successfully", projects=[result.data[0]])
    except Exception as e:
        return ProjectResponse(success=False, message=str(e))
    

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
        topic = extract_topic_from_script(script).topic
        result = supabase.table("scripts").insert({"project_id": project_id, "script": script, "topic": topic}).execute()
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
        project_id = request.project_id
        script = write_script.main(
            project_id=project_id, 
            title=request.topic,
            context=request.context, 
            channel_number=request.channel_number,
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
        result = supabase.table("scripts").select("id, title, topic, script, created_at").filter("project_id", "eq", project_id).execute()
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
        print(result.data)
        return ScriptResponse(success=True, message="Script property fetched successfully", scripts=result.data)
    except Exception as e:
        return ScriptResponse(success=False, message=str(e))


@app.post("/voice-over", response_model=VoiceoverResponse)
def generate_voice_over(request: VoiceoverRequest, current_user: dict = Depends(get_current_user)):
    try:
        
        script_id = request.script_id
        # Script'i güvenli bir şekilde ele al
        project_id = request.project_id
        try:
            # Sanitizer middleware ile işlenmemiş olması durumunda manuel olarak sanitize et
            from api.security.sanitizer import sanitize_input
            project_id = sanitize_input(project_id,script_id, context="script")
        except Exception as e:
            print(f"Script sanitize hatası: {str(e)}")
        
        channel_number = request.channel_number
        voice_over_url = voice_over.main(project_id,script_id, channel_number)
        
        return VoiceoverResponse(success=True, message="Voice over generated successfully", voice_over_url=voice_over_url)
    except Exception as e:
        return VoiceoverResponse(success=False, message=str(e)), 500


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
        voice_over_id = request.voice_over_id
        captions.main(project_id, voice_over_id, request.channel_number)
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
            return VideoEditResponse(success=False, message="Video editing or upload process failed"), 500
            
    except AttributeError:
        # Eğer request.script_id mevcut değilse bu hata alınabilir
        return VideoEditResponse(success=False, message="Missing 'script_id' in request body.")
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        # Hata detaylarını loglamak iyi bir pratik olabilir
        print(f"Video editing endpoint error: {str(e)}\n{error_details}")
        return VideoEditResponse(success=False, message=f"An error occurred during the video editing request: {str(e)}")
    

@app.get("/response-json/{project_id}", response_model=ResponseJson)
def get_response_json(project_id: int):
    data = supabase.table("projects").select("response_json").eq("id", project_id).execute()
    json_data = json.loads(data.data[0]["response_json"])
    for clip_data in json_data:
        signed_url_raw = supabase.storage.from_("video-database").create_signed_url(clip_data["clip_name"], 3600)
        signed_url = signed_url_raw.get('signedURL')
        clip_data["clip_url"] = signed_url
    print(json_data)
    if data.data is None:
        return ResponseJson(success=False, message="Response JSON not found")
    return ResponseJson(success=True, message="Response JSON fetched successfully", data=json_data)

@app.get("/storyboards", response_model=StoryboardResponse)
def get_storyboards(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        result = supabase.table("storyboards").select("id, project_id, name, initial_images_created, created_at, updated_at").eq("user_id", user_id).execute()
        return StoryboardResponse(success=True, message="Storyboards fetched successfully", storyboards=result.data)
    except Exception as e:
        return StoryboardResponse(success=False, message=str(e), storyboards=[])


@app.get("/storyboards/{storyboard_id}", response_model=StoryboardResponse)
def get_storyboard(storyboard_id: str, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        
        print(user_id)
        # String ID'yi integer'a çevir
        storyboard_id_int = int(storyboard_id)
        print(storyboard_id)
        result = supabase.table("storyboards").select("id, project_id, name, story_board, initial_images_created, created_at, updated_at").eq("user_id", user_id).eq("id", storyboard_id_int).execute()
        
        if not result.data:
            return StoryboardResponse(success=False, message="Storyboard not found", storyboards=[])
            
        return StoryboardResponse(success=True, message="Storyboard fetched successfully", storyboards=[result.data[0]])
    except ValueError:
        return StoryboardResponse(success=False, message="Invalid storyboard ID format", storyboards=[])
    except Exception as e:
        print(f"Error in get_storyboard: {str(e)}")  # Hata loglaması ekle
        return StoryboardResponse(success=False, message=str(e), storyboards=[])

@app.post("/storyboard", response_model=StoryboardResponse)
def create_storyboard(request: StoryboardRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        result = supabase.table("storyboards").insert({
            "user_id": user_id,
            "project_id": request.project_id,
            "name": request.name,
            "story_board": request.story_board,
            "initial_images_created": request.initial_images_created
        }).execute()
        return StoryboardResponse(success=True, message="Storyboard created successfully", storyboards=[result.data[0]])
    except Exception as e:
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
            aspect_ratio = str(width / height) if height != 0 else "0" # Sıfıra bölme hatasını engelle
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


        result = supabase.table("uploaded_images").insert({
            "user_id": user_id,
            "url": image_public_url, 
            "image_path": image_path,
            "aspect_ratio": aspect_ratio,
            "name": request.name
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
def generate_single_video(request: GenerateSingleVideoRequest, current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        model = request.model
        video = None


        match model:
            case "klingai":
                video = generate_klingAI_video(request.prompt, request.negativePrompt, request.duration, request.cfgScale, request.aspectRatio, request.startImage, request.endImage)
            case "runwayml":
                video = generate_runwayML_video(request.prompt, request.negativePrompt, request.duration, request.cfgScale, request.aspectRatio, request.startImage, request.endImage)
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
            "url": video_url,
            "name": video_name
        }).execute()


        video_id = video_insert_result.data[0]["id"]

        update_start_image_result = supabase.table("uploaded_images").update({
            "video_id": video_id
        }).eq("id", request.startImageId).execute()

        if request.endImageId:
            update_end_image_result = supabase.table("uploaded_images").update({
                "video_id": video_id
            }).eq("id", request.endImageId).execute()

        return GenerateSingleVideoResponse(success=True, message="Video generated successfully", video=video_url)
    except Exception as e:
        return GenerateSingleVideoResponse(success=False, message=str(e))


@app.post("/generate-timeline-video")
def generate_timeline_video(request: GenerateTimelineVideoRequest, current_user: dict = Depends(get_current_user)):
    try:
        videos = []
        user_id = current_user["user_id"]
        model = request.model
        video = None
        for segment in request.segments:
            match model:
                case "klingai":
                    video = generate_klingAI_video(segment.prompt, None, segment.duration, segment.cfg_scale, request.aspect_ratio, segment.start_image, segment.end_image)
                case "runwayml":
                    video = generate_runwayML_video(segment.prompt, None, segment.duration, segment.cfg_scale, request.aspect_ratio, segment.start_image, segment.end_image)
            if video is None:
                return GenerateTimelineVideoResponse(success=False, message="Video generation failed")
            
            video_name = f"videos/{datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}.mp4"
            video_upload_result = supabase.storage.from_("videos").upload(path=video_name, file=video)
            video_url = supabase.storage.from_("videos").get_public_url(video_name)
            videos.append(video_url)
            video_insert_result = supabase.table("generated_videos").insert({
                "user_id": user_id,
                "url": video_url,
                "name": video_name
            }).execute()
            video_id = video_insert_result.data[0]["id"]
            for segment in request.segments:
                update_image_result_start = supabase.table("uploaded_images").update({
                    "video_id": video_id
                }).eq("id", segment.start_image_id).execute()
                update_image_result_end = supabase.table("uploaded_images").update({
                    "video_id": video_id
                }).eq("id", segment.end_image_id).execute()


        return GenerateTimelineVideoResponse(success=True, message="Video generated successfully", video_urls=videos,segments=request.segments)
        
    except Exception as e:
        return GenerateTimelineVideoResponse(success=False, message=str(e))

@app.get("/video-list", response_model=VideoListResponse)
def get_video_list(current_user: dict = Depends(get_current_user)):
    try:
        user_id = current_user["user_id"]
        # user_id'ye göre sadece url'leri seç
        result = supabase.table("generated_videos").select("id,url,created_at").eq("user_id", user_id).order("created_at", desc=True).execute()

        if result.data is not None:
            # Eğer result.data boş bir liste değilse ve içinde öğeler varsa
            if result.data: 
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
