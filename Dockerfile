# --- Aşama 1: Base ---
# Bu aşama, hem API hem de Worker için ortak olan tüm bağımlılıkları kurar.
FROM python:3.10.11-slim AS base

WORKDIR /app

# Ortak sistem bağımlılıklarını yükle
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Tüm Python gereksinimlerini yükle
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Uygulama çalışırken kullanılacak ortam değişkenlerini tanımla
ENV PYTHONPATH="/app:${PYTHONPATH}"

# --- Aşama 2: Worker ---
# 'base' aşamasından başlar ve sadece worker için gerekli dosyaları ekler.
FROM base AS worker

# Sadece worker'ın ihtiyaç duyduğu dosyaları kopyala
COPY tasks.py .
COPY video_edit.py .
COPY file_manager.py .
COPY config.py .
COPY captions.py .
COPY auto_editor/ /app/auto_editor/
COPY logging_system/ /app/logging_system/
COPY perf_render_timeline.py .
COPY api/websockets/pubsub.py /app/api/websockets/pubsub.py

# Worker için varsayılan komut
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]


# --- Aşama 3: API (Son aşama olduğu için varsayılan olarak bu kullanılır) ---
# 'base' aşamasından başlar ve API için gerekli dosyaları ekler.
FROM base AS api

# Worker'ın dosyaları dahil tüm API kodunu kopyala (API, görevleri tetiklemek için tasks.py'yi bilmelidir)
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

# API'nin çalışacağı portu belirle
EXPOSE 8000

# API için varsayılan komut
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
