FROM python:3.10.11-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Gereksinimleri kopyala ve yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama dosyalarını kopyala
COPY api/ /app/api/
COPY write_script.py .
COPY voice_over.py .
COPY file_manager.py .
COPY config.py .
COPY captions.py .
COPY video_edit.py .
COPY timeline_manager.py .
COPY auto_editor/ /app/auto_editor/
COPY logging_system/ /app/logging_system/
COPY perf_render_timeline.py .
COPY tasks.py .

# Gerekli dizinleri oluştur
RUN mkdir -p /app/context
RUN mkdir -p /app/api/ResponseSchemes
RUN mkdir -p /app/api/RequestSchemes
RUN mkdir -p /app/api/security
RUN mkdir -p /app/api/auth
RUN mkdir -p /app/api/db


# Örnek script dosyasını oluştur veya kopyala
RUN echo '<Example 1>' > /app/amazing_script.txt
RUN echo 'Memory content goes here' > /app/context/memory.txt

# Uygulama çalışırken kullanılacak ortam değişkenlerini tanımla
ENV PYTHONPATH="/app:${PYTHONPATH}"

# API'nin çalışacağı portu belirle
EXPOSE 8000

# Uygulamayı çalıştır
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
