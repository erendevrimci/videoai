import os
import jwt
from fastapi import Depends, HTTPException, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()
# Güvenlik şeması
security = HTTPBearer()
supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_KEY")
jwt_secret = os.environ.get("SUPABASE_JWT_SECRET")  # Supabase JWT secret

# Token doğrulama fonksiyonu
def verify_jwt(credentials: HTTPAuthorizationCredentials = Security(security)):
    token = credentials.credentials
    
    
    try:
        # JWT token'ı doğrula - audience parametresi eklendi
        payload = jwt.decode(
            token, 
            jwt_secret, 
            algorithms=["HS256"],
            audience="authenticated",  # Supabase token'larında audience değeri
            options={"verify_signature": True}
        )
        
        # Ham token'ı payload içine ekle, böylece get_current_user'a aktarılabilir
        payload["_raw_token"] = token
        
        
        return payload
    except jwt.PyJWTError as e:
        # Hata detayını logla
        print(f"JWT Doğrulama hatası: {str(e)}")
        raise HTTPException(status_code=401, detail=f"Geçersiz token: {str(e)}")

# Kullanıcı kimliği doğrulama
def get_current_user(payload: dict = Depends(verify_jwt)):
    # Token payload'ından kullanıcı bilgilerini al
    try:
        # Supabase'de user_id olarak "sub" alanı kullanılıyor
        user_id = payload.get("sub")
        email = payload.get("email")
        
        if not user_id:
            raise HTTPException(status_code=401, detail="Kullanıcı kimliği bulunamadı")
        
        # Ham token'ı al
        raw_token = payload.get("_raw_token")
        
        # Kullanıcı bilgilerini içeren bir sözlük döndür
        return {
            "user_id": user_id,
            "email": email,
            "role": payload.get("role"),
            "app_metadata": payload.get("app_metadata"),
            "user_metadata": payload.get("user_metadata"),
            "token": raw_token  # Ham JWT token'ı ekle
        }
    except Exception as e:
        print(f"Kullanıcı bilgisi alma hatası: {str(e)}")
        raise HTTPException(status_code=401, detail="Kullanıcı bilgisi alınamadı")