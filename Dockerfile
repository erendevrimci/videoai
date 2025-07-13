# --- Worker Build ---
FROM python:3.10.11-slim

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

# Root olmayan bir kullanıcı oluştur
RUN addgroup --system appuser && adduser --system --ingroup appuser --no-create-home appuser

# Sadece worker'ın ihtiyaç duyduğu dosyaları kopyala
COPY tasks.py .
COPY video_edit.py .
COPY file_manager.py .
COPY config.py .
COPY captions.py .
COPY timeline_manager.py .
COPY auto_editor/ /app/auto_editor/
COPY logging_system/ /app/logging_system/
COPY perf_render_timeline.py .
COPY api/websockets/pubsub.py /app/api/websockets/pubsub.py
# pubsub.py'nin çalışması için boş dizinler gerekebilir
RUN mkdir -p /app/api/websockets/

# Dizinin sahipliğini yeni kullanıcıya ver
RUN chown -R appuser:appuser /app

# Root olmayan kullanıcıya geç
USER appuser

# Worker için varsayılan komut
CMD ["celery", "-A", "tasks", "worker", "--loglevel=info"]
