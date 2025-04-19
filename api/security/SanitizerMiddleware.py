from starlette.middleware.base import BaseHTTPMiddleware
from fastapi import Request, Response, status
from fastapi.responses import JSONResponse
from .sanitizer import sanitize_dict, sanitize_input, SQLInjectionError
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
                            original_script_content = None
                            original_context_content = None
                            is_special_content = False

                            if isinstance(json_body, dict):
                                if "script" in json_body:
                                    is_special_content = True
                                    original_script_content = json_body.pop("script", None)
                                    security_logger.debug("Script alanı sanitizasyon için geçici olarak çıkarıldı.")

                                if "context" in json_body:
                                    is_special_content = True
                                    original_context_content = json_body.pop("context", None)
                                    security_logger.debug("Context alanı sanitizasyon için geçici olarak çıkarıldı.")

                            sanitized_body = sanitize_dict(
                                json_body,
                                ip_address=client_ip,
                                is_script_content=is_special_content
                            )

                            if original_script_content is not None:
                                sanitized_body["script"] = original_script_content
                                security_logger.debug("Orijinal script alanı sanitize edilmiş body'e geri eklendi.")

                            if original_context_content is not None:
                                sanitized_body["context"] = original_context_content
                                security_logger.debug("Orijinal context alanı sanitize edilmiş body'e geri eklendi.")

                            request._body = json.dumps(sanitized_body).encode()
                            security_logger.debug(f"Sanitize edilmiş body request'e yazıldı: {sanitized_body}")

                        except SQLInjectionError as e:
                            security_logger.warning(f"SQL Injection girişimi (JSON): {str(e)}, IP: {client_ip}, Body (kısmi): {str(body)[:200]}")
                            return Response(
                                content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                                status_code=status.HTTP_403_FORBIDDEN,
                                media_type="application/json"
                            )
                except json.JSONDecodeError:
                    security_logger.warning(f"Geçersiz JSON formatı: {request.method} {request.url.path} - IP: {client_ip}")
                    pass
                except Exception as e:
                    security_logger.error(f"JSON body işlerken beklenmedik hata: {str(e)} - IP: {client_ip}", exc_info=True)
            
            # 2. PATH parametrelerini sanitize et (tüm istekler için)
            if request.path_params:
                try:
                    sanitized_path_params = {}
                    security_logger.debug(f"PATH params: {request.path_params}")
                    
                    for key, value in request.path_params.items():
                        if isinstance(value, str):
                            sanitized_value = sanitize_input(value, context="general", ip_address=client_ip)
                            sanitized_path_params[key] = sanitized_value
                            security_logger.debug(f"Sanitized PATH param {key}: '{value}' -> '{sanitized_value}'")
                        else:
                            sanitized_path_params[key] = value
                    
                    setattr(request.state, "sanitized_path_params", sanitized_path_params)
                    security_logger.debug(f"Sanitized PATH params added to state: {sanitized_path_params}")
                except SQLInjectionError as e:
                    security_logger.warning(f"SQL Injection girişimi (PATH): {str(e)}, IP: {client_ip}")
                    return Response(
                        content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                        status_code=status.HTTP_403_FORBIDDEN,
                        media_type="application/json"
                    )
                except Exception as e:
                    security_logger.error(f"Path parametrelerini işlerken hata: {str(e)} - IP: {client_ip}", exc_info=True)
            
            # 3. QUERY parametrelerini sanitize et (tüm istekler için)
            if request.query_params:
                try:
                    sanitized_query_params = {}
                    for key, value in request.query_params.items():
                        if isinstance(value, str):
                            context_type = "search" if key in ["q", "query", "search", "keyword"] else "general"
                            sanitized_query_params[key] = sanitize_input(value, context=context_type, ip_address=client_ip)
                        else:
                            sanitized_query_params[key] = value
                    
                    setattr(request.state, "sanitized_query_params", sanitized_query_params)
                    security_logger.debug(f"Sanitized QUERY params added to state: {sanitized_query_params}")
                except SQLInjectionError as e:
                    security_logger.warning(f"SQL Injection girişimi (QUERY): {str(e)}, IP: {client_ip}")
                    return Response(
                        content=json.dumps({"detail": "Güvenlik ihlali tespit edildi"}),
                        status_code=status.HTTP_403_FORBIDDEN,
                        media_type="application/json"
                    )
                except Exception as e:
                    security_logger.error(f"Query parametrelerini işlerken hata: {str(e)} - IP: {client_ip}", exc_info=True)
            
            # 4. Dependency Injection için bir sanitizasyon dependency ekle
            setattr(request.state, "sanitize_input", sanitize_input)
            security_logger.debug("Sanitize_input function added to state")
        
        except Exception as e:
            security_logger.error(f"Middleware genel hata: {str(e)} - IP: {client_ip}", exc_info=True)
        
        # İşleme devam et
        response = await call_next(request)
        return response

