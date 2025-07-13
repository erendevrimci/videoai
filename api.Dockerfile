# --- API Build ---
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

# API için gerekli dosyaları kopyala
# API, görevleri tetiklemek için tasks.py'yi bilmelidir
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
# Not: COPY komutları bu dizinleri zaten oluşturabilir, ancak yine de ekliyoruz.
RUN mkdir -p /app/context \
    /app/api/ResponseSchemes \
    /app/api/RequestSchemes \
    /app/api/security \
    /app/api/auth \
    /app/api/db

# Örnek dosyaları oluştur
RUN echo '<Example 1>' > /app/amazing_script.txt
RUN echo 'Memory content goes here' > /app/context/memory.txt

# Dizinin sahipliğini yeni kullanıcıya ver
RUN chown -R appuser:appuser /app

# Root olmayan kullanıcıya geç
USER appuser

# API'nin çalışacağı portu belirle
EXPOSE 8000

# API için varsayılan komut
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"] 