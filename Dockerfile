# ... existing code ...
# Sadece worker'ın ihtiyaç duyduğu dosyaları kopyala
COPY tasks.py .
COPY video_edit.py .
COPY file_manager.py .
COPY config.py .
COPY captions.py .
COPY timeline_manager.py .
COPY auto_editor/ /app/auto_editor/
COPY logging_system/ /app/logging_system/
# ... existing code ...
