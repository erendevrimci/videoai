from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Response, Request, status
import time
import json
from collections import defaultdict
import logging
import asyncio

# Rate limiter için logger
rate_limiter_logger = logging.getLogger("rate_limiter")

class RateLimiterMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, 
                 limit_per_minute=60,       # Dakikada maksimum istek sayısı
                 limit_per_hour=1000,       # Saatte maksimum istek sayısı
                 whitelist_paths=None,      # Rate limit'ten muaf tutulacak path'ler
                 blacklist_ips=None):       # Tamamen engellenen IP'ler
        super().__init__(app)
        self.limit_per_minute = limit_per_minute
        self.limit_per_hour = limit_per_hour
        self.whitelist_paths = whitelist_paths or ["/health", "/docs", "/redoc", "/openapi.json"]
        self.blacklist_ips = blacklist_ips or []
        
        # İstek sayaçları - üretim ortamında Redis gibi bir depolama kullanılmalıdır
        self.request_counts = defaultdict(lambda: {"minute": [], "hour": []})
        
        # Her 10 dakikada bir eski kayıtları temizle
        self.cleanup_task = asyncio.create_task(self._cleanup_old_records())
    
    async def _cleanup_old_records(self):
        """Eski istek kayıtlarını temizler"""
        while True:
            try:
                current_time = time.time()
                # Tüm IP'ler için döngü
                for ip, records in list(self.request_counts.items()):
                    # Bir saatten eski kayıtları temizle
                    records["minute"] = [t for t in records["minute"] if current_time - t < 60]
                    records["hour"] = [t for t in records["hour"] if current_time - t < 3600]
                    
                    # Boş kayıtları tamamen sil
                    if not records["minute"] and not records["hour"]:
                        del self.request_counts[ip]
            except Exception as e:
                rate_limiter_logger.error(f"Temizleme hatası: {str(e)}")
            
            # 10 dakika bekle
            await asyncio.sleep(600)
    
    async def dispatch(self, request: Request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        path = request.url.path
        
        # Whitelist kontrolü
        if any(path.startswith(wpath) for wpath in self.whitelist_paths):
            return await call_next(request)
        
        # Blacklist kontrolü
        if client_ip in self.blacklist_ips:
            rate_limiter_logger.warning(f"Blacklist IP erişim girişimi: {client_ip}, Path: {path}")
            return Response(
                content=json.dumps({"detail": "Erişim engellendi"}),
                status_code=status.HTTP_403_FORBIDDEN,
                media_type="application/json"
            )
        
        # Şu anki zaman
        current_time = time.time()
        
        # İstek sayılarını güncelle
        self.request_counts[client_ip]["minute"].append(current_time)
        self.request_counts[client_ip]["hour"].append(current_time)
        
        # Son 1 dakikadaki istekleri hesapla
        minute_count = len([t for t in self.request_counts[client_ip]["minute"] 
                         if current_time - t < 60])
        
        # Son 1 saatteki istekleri hesapla
        hour_count = len([t for t in self.request_counts[client_ip]["hour"] 
                       if current_time - t < 3600])
        
        # Dakika limitini kontrol et
        if minute_count > self.limit_per_minute:
            rate_limiter_logger.warning(
                f"Rate limit aşıldı (dakika): IP: {client_ip}, Count: {minute_count}"
            )
            return Response(
                content=json.dumps({
                    "detail": "Çok fazla istek yapıldı. Lütfen bir dakika bekleyin."
                }),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "60"},
                media_type="application/json"
            )
        
        # Saat limitini kontrol et
        if hour_count > self.limit_per_hour:
            rate_limiter_logger.warning(
                f"Rate limit aşıldı (saat): IP: {client_ip}, Count: {hour_count}"
            )
            return Response(
                content=json.dumps({
                    "detail": "Saatlik istek limitinizi aştınız. Lütfen daha sonra tekrar deneyin."
                }),
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                headers={"Retry-After": "3600"},
                media_type="application/json"
            )
        
        # Normal işleme devam et
        return await call_next(request)

