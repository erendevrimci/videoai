import os
import asyncio
import time
from typing import Optional
from uuid import uuid4

# KULLANICI NOTU: Bu fonksiyonun çalışması için aşağıdaki paketlerin yüklenmesi gerekmektedir:
# pip install google-genai google-cloud-storage
try:
    import google.genai as genai
    from google.genai.types import GenerateVideosConfig
except ImportError:
    raise ImportError("Lütfen Veo için gerekli paketleri kurun: pip install google-genai google-cloud-storage")

# Referans: https://cloud.google.com/vertex-ai/generative-ai/docs/video/generate-videos-from-text

# --- ORTAM DEĞİŞKENLERİ VE KİMLİK DOĞRULAMA ---
# Bu kodun çalışabilmesi için aşağıdaki ortam değişkenlerinin ayarlandığından emin olun:
# 1. GOOGLE_CLOUD_PROJECT: Sizin Google Cloud Proje ID'niz.
# 2. GCP_VIDEO_OUTPUT_BUCKET: Üretilen videoların yükleneceği GCS bucket adı.
# 3. GOOGLE_CLOUD_LOCATION: "us-central1" gibi projenizin konumu.
# 4. GOOGLE_GENAI_USE_VERTEXAI: Vertex AI kullanılacağını belirtir, kod tarafından "True" olarak ayarlanır.
# Kimlik doğrulaması için `gcloud auth application-default login` komutunu çalıştırmanız önerilir.
# ----------------------------------------------------
try:
    if not os.getenv("GOOGLE_GENAI_USE_VERTEXAI"):
        os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "True"

    PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
    GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
    if not PROJECT_ID :
        raise ValueError("GOOGLE_CLOUD_PROJECT ortam değişkeni ayarlanmalıdır.")

    # genai.configure() SDK'nın en son sürümlerinde gerekli olmayabilir,
    # Client() başlatma sırasında proje ve konumu otomatik olarak alabilir.
    # Ancak yine de belirtmekte fayda var.
    print(f"Vertex AI için GenAI SDK'sı {PROJECT_ID} projesi ile yapılandırılıyor.")

    # İstemciyi modul seviyesinde bir kere başlatarak yeniden kullanımını sağlıyoruz.
    # Bu, 'NoneType' object is not callable' gibi başlatma hatalarını önleyebilir.
    client = genai.Client(vertexai=True, project=PROJECT_ID, location=GOOGLE_CLOUD_LOCATION)

except Exception as e:
    raise RuntimeError(f"Vertex AI (GenAI) istemcisi başlatılamadı. Gerekli ortam değişkenlerini kontrol edin. Hata: {e}")


async def generate_veo3_video(
    prompt: str,
    duration: int = 8,
    aspect_ratio: str = "9:16",
    negative_prompt: Optional[str] = None,
    sampleCount: int = 1,
) -> bytes | None:
   
    print(f"Veo | Segment '{prompt[:30]}...' için üretim başlıyor.")

    # client = genai.Client(vertexai=True, project=PROJECT_ID, location=GOOGLE_CLOUD_LOCATION)
    
    print(f"Veo | Model `veo-3.0-generate-preview` kullanılıyor.")
   
    try:
        config = GenerateVideosConfig(
            aspect_ratio=aspect_ratio,
            negative_prompt=negative_prompt,
            duration_seconds=duration,
        )

        operation = client.models.generate_videos(
            model="veo-3.0-generate-preview",
            prompt=prompt,
            config=config,
        )
        print(f"Veo | Üretim işlemi başlatıldı. Operasyon adı: {operation.name}")
        print("Veo | Video üretiliyor, bu işlem birkaç dakika sürebilir...")

        loop = asyncio.get_running_loop()

        # Asenkron ve doğru polling (durum kontrolü) döngüsü
        while not operation.done:
            await asyncio.sleep(15) # Programı bloklamadan bekle
            print("Veo | Durum kontrol ediliyor...")
            # client.operations.get senkron bir I/O çağrısı olduğu için
            # asyncio event loop'unu bloklamamak adına executor içinde çalıştırılır.
            operation = await loop.run_in_executor(
                None,
                # operation.name, operasyonun tam adını içerir
                lambda: client.operations.get(operation=operation)
            )

        print("Veo | İşlem tamamlandı.")

        # Operasyon bittiğinde, hata olup olmadığını kontrol et
        if operation.error:
            raise Exception(f"Video üretiminde hata oluştu: {operation.error.message}")

        # .result bir niteliktir (attribute), .result() gibi bir metod değil
        response = operation.result

        if not response:
             raise Exception("Video üretimi tamamlandı ancak bir yanıt alınamadı.")

        video_generated_bytes = response.generated_videos[0].video.video_bytes
        
        return video_generated_bytes

    except Exception as e:
        print(f"HATA: Veo | Video oluşturma hatası: {e}")
        raise e
