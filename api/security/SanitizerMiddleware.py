from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, status
from security.sanitizer import sanitize_dict, sanitize_input, SQLInjectionError
import json
import logging

# Güvenlik logları için logger
security_logger = logging.getLogger("security")

class SanitizerMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request, call_next):
        client_ip = request.client.host if request.client else "unknown"
        
        # İstek bilgilerini logla
        security_logger.debug(f"İstek: {request.method} {request.url.path} - IP: {client_ip}")
        
        try:
            # 1. POST, PUT, PATCH istekleri için body sanitizasyonu
            if request.method in ["POST", "PUT", "PATCH"]:
                try:
                    body = await request.body()
                    if body:
                        json_body = json.loads(body)
                        try:
                            # Script içeren istek mi kontrol et
                            is_script_content = False
                            script_content = None
                            
                            # /voice-over endpoint'i veya "script" alanı içeren JSON için özel işlem
                            if request.url.path == "/voice-over" and isinstance(json_body, dict) and "script" in json_body:
                                is_script_content = True
                                script_content = json_body["script"]
                            
                            # JSON verisini sanitize et
                            sanitized_body = sanitize_dict(
                                json_body, 
                                ip_address=client_ip, 
                                is_script_content=is_script_content
                            )
                            
                            # Script içeriğini geri koyalım (özel olarak işlendiği için)
                            if is_script_content and script_content:
                                sanitized_body["script"] = script_content
                                
                            # Request nesnesini güncelle
                            request._body = json.dumps(sanitized_body).encode()
                        except SQLInjectionError as e:
                            security_logger.warning(f"SQL Injection girişimi (JSON): {str(e)}, IP: {client_ip}")
                            return Response(
                                content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                                status_code=status.HTTP_403_FORBIDDEN,
                                media_type="application/json"
                            )
                except json.JSONDecodeError:
                    # JSON çözümleme hatası veya boş body - devam et
                    pass
                except Exception as e:
                    security_logger.error(f"JSON body işlerken hata: {str(e)}")
            
            # 2. PATH parametrelerini sanitize et (tüm istekler için)
            # FastAPI'de path_params doğrudan değiştirilemez, bu nedenle state'e ekliyoruz
            if request.path_params:
                try:
                    sanitized_path_params = {}
                    security_logger.debug(f"PATH params: {request.path_params}")
                    
                    for key, value in request.path_params.items():
                        if isinstance(value, str):
                            # Path parametresi için sanitize işlemi
                            sanitized_value = sanitize_input(value, context="general", ip_address=client_ip)
                            sanitized_path_params[key] = sanitized_value
                            security_logger.debug(f"Sanitized PATH param {key}: '{value}' -> '{sanitized_value}'")
                        else:
                            sanitized_path_params[key] = value
                    
                    # Sanitize edilmiş parametreleri state'e ekle
                    setattr(request.state, "sanitized_path_params", sanitized_path_params)
                    security_logger.info(f"Sanitized PATH params added to state: {sanitized_path_params}")
                    # State içeriğini kontrol et
                    security_logger.info(f"State içeriği (path eklendikten sonra): {dir(request.state)}")
                except SQLInjectionError as e:
                    security_logger.warning(f"SQL Injection girişimi (PATH): {str(e)}, IP: {client_ip}")
                    return Response(
                        content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                        status_code=status.HTTP_403_FORBIDDEN,
                        media_type="application/json"
                    )
                except Exception as e:
                    security_logger.error(f"Path parametrelerini işlerken hata: {str(e)}")
            
            # 3. QUERY parametrelerini sanitize et (tüm istekler için)
            if request.query_params:
                try:
                    sanitized_query_params = {}
                    for key, value in request.query_params.items():
                        if isinstance(value, str):
                            # Arama sorguları için farklı sanitizasyon kuralları
                            context = "search" if key in ["q", "query", "search", "keyword"] else "general"
                            sanitized_query_params[key] = sanitize_input(value, context=context, ip_address=client_ip)
                        else:
                            sanitized_query_params[key] = value
                    
                    # Sanitize edilmiş parametreleri state'e ekle
                    setattr(request.state, "sanitized_query_params", sanitized_query_params)
                    security_logger.info(f"Sanitized QUERY params added to state: {sanitized_query_params}")
                except SQLInjectionError as e:
                    security_logger.warning(f"SQL Injection girişimi (QUERY): {str(e)}, IP: {client_ip}")
                    return Response(
                        content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                        status_code=status.HTTP_403_FORBIDDEN,
                        media_type="application/json"
                    )
                except Exception as e:
                    security_logger.error(f"Query parametrelerini işlerken hata: {str(e)}")
            
            # 4. Dependency Injection için bir sanitizasyon dependency ekle
            # Bu, endpoint fonksiyonlarında parametre olarak kullanılabilir
            setattr(request.state, "sanitize_input", sanitize_input)
            security_logger.info("Sanitize_input function added to state")
        
        except Exception as e:
            security_logger.error(f"Middleware genel hata: {str(e)}")
        
        # İşleme devam et
        response = await call_next(request)
        return response

