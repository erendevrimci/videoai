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
                            original_story_board_segments = [] # (index, segment_text) saklamak için

                            if isinstance(json_body, dict):
                                if "script" in json_body:
                                    original_script_content = json_body.pop("script", None)
                                    security_logger.debug("Ana 'script' alanı sanitizasyon için geçici olarak çıkarıldı.")

                                if "context" in json_body:
                                    original_context_content = json_body.pop("context", None)
                                    security_logger.debug("Ana 'context' alanı sanitizasyon için geçici olarak çıkarıldı.")

                                if "story_board" in json_body and isinstance(json_body["story_board"], list):
                                    security_logger.debug("'story_board' alanı bulundu. İçindeki 'scriptSegment' alanları işlenecek.")
                                    # Listenin kopyası üzerinde iterasyon yapmıyoruz, doğrudan değiştiriyoruz.
                                    # Orijinal indexleri korumak için dikkatli olmalıyız.
                                    for i, item in enumerate(list(json_body["story_board"])): # Geçici kopya üzerinde iterasyon
                                        if isinstance(item, dict) and "scriptSegment" in item:
                                            segment = item.pop("scriptSegment", None) # Orijinal item'dan çıkar
                                            if segment is not None:
                                                original_story_board_segments.append((i, segment))
                                                security_logger.debug(f"  'story_board' [{i}] içindeki 'scriptSegment' geçici olarak çıkarıldı.")
                                    # json_body["story_board"] şimdi scriptSegment'leri çıkarılmış item'ları içeriyor.

                            # 'script', 'context' ve 'scriptSegment'ler çıkarılmış body'i sanitize et
                            sanitized_body = sanitize_dict(
                                json_body,
                                ip_address=client_ip,
                                is_script_content=False # Script benzeri içerikleri manuel ele aldığımız için False
                            )

                            # Ana 'script' alanını sanitize et ve geri ekle
                            if original_script_content is not None:
                                sanitized_body["script"] = sanitize_input(original_script_content, context="script", ip_address=client_ip)
                                security_logger.debug("Sanitize edilmiş ana 'script' alanı geri eklendi.")

                            # Ana 'context' alanını sanitize et ve geri ekle
                            if original_context_content is not None:
                                sanitized_body["context"] = sanitize_input(original_context_content, context="script", ip_address=client_ip)
                                security_logger.debug("Sanitize edilmiş ana 'context' alanı geri eklendi.")

                            # 'scriptSegment'leri sanitize et ve 'story_board'a geri ekle
                            if "story_board" in sanitized_body and isinstance(sanitized_body["story_board"], list):
                                if original_story_board_segments:
                                    security_logger.debug("Sanitize edilmiş 'scriptSegment'ler 'story_board'a geri ekleniyor.")
                                for index, segment_text in original_story_board_segments:
                                    if 0 <= index < len(sanitized_body["story_board"]):
                                        if isinstance(sanitized_body["story_board"][index], dict):
                                            sanitized_segment = sanitize_input(segment_text, context="script", ip_address=client_ip)
                                            sanitized_body["story_board"][index]["scriptSegment"] = sanitized_segment
                                            security_logger.debug(f"  Sanitize edilmiş 'scriptSegment' 'story_board' [{index}] içine geri eklendi.")
                                        else:
                                            security_logger.warning(f"  'story_board' [{index}] bir sözlük değil. 'scriptSegment' geri eklenemedi.")
                                    else:
                                        security_logger.warning(f"  'story_board' için index ({index}) sınır dışında. 'scriptSegment' geri eklenemedi: {segment_text[:50]}...")
                            elif original_story_board_segments:
                                security_logger.warning("'story_board' alanı sanitize_dict tarafından değiştirildi/kaldırıldı. Tüm 'scriptSegment'ler geri eklenemeyebilir.")
                            
                            request._body = json.dumps(sanitized_body).encode()
                            security_logger.debug(f"Sanitize edilmiş body request'e yazıldı (kısmi): {str(sanitized_body)[:500]}")

                        except SQLInjectionError as e:
                            security_logger.warning(f"SQL Injection girişimi (JSON): {str(e)}, IP: {client_ip}, Body (kısmi): {str(body)[:200]}")
                            return JSONResponse( # FastAPI'nin JSONResponse'unu kullanmak daha iyi olabilir
                                status_code=status.HTTP_403_FORBIDDEN,
                                content={"detail": "Güvenlik ihlali tespit edildi"}
                            )
                except json.JSONDecodeError:
                    security_logger.warning(f"Geçersiz JSON formatı: {request.method} {request.url.path} - IP: {client_ip}")
                    # Hata döndürmek yerine isteğin devam etmesine izin verilebilir (call_next), 
                    # veya özel bir yanıt döndürülebilir. Mevcut davranış 'pass' idi.
                    # Eğer 'pass' ise ve body yoksa veya hatalıysa, sonraki katmanlar bunu ele almalı.
                    # Ancak burada bir JSON decode hatası sonrası genellikle 400 Bad Request dönmek daha iyidir.
                    # Şimdilik orijinal 'pass' davranışını koruyalım ama not olarak kalsın.
                    pass # pass
                except Exception as e:
                    security_logger.error(f"JSON body işlerken beklenmedik hata: {str(e)} - IP: {client_ip}", exc_info=True)
                    # Genel bir sunucu hatası olarak ele alınabilir
                    return JSONResponse(
                        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                        content={"detail": "İstek işlenirken sunucu tarafında bir hata oluştu."}
                    )
            
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
            # Bu genel hata da 500 olarak döndürülebilir
            return JSONResponse(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                content={"detail": "Middleware işlenirken bir hata oluştu."}
            )
        
        response = await call_next(request)
        return response

