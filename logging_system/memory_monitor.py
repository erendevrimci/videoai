import os
import psutil
import logging

def log_memory_usage(label: str):
    """
    Mevcut Python işleminin RAM kullanımını belirli bir etiketle loglar.
    """
    process = psutil.Process(os.getpid())
    memory_mb = process.memory_info().rss / (1024 * 1024)
    logging.info(f"[MemoryLog] {label}: {memory_mb:.2f} MB") 