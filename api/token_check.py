from supabase import create_client, Client
from dotenv import load_dotenv
import os
import jwt
from jose import JWTError, jwt as jose_jwt

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
jwt_secret = os.getenv("SUPABASE_JWT_SECRET")  

def verify_token(token):
    print("Token kontrolü yapılıyor...")
    try:
        # PyJWT ile kontrol - audience parametresi eklendi
        try:
            payload = jwt.decode(
                token, 
                jwt_secret, 
                algorithms=["HS256"],
                audience="authenticated",  # 'aud' parametresi token içeriğinden alındı
                options={"verify_signature": True}
            )
            print("PyJWT ile token doğrulandı:", payload)
            return True, payload
        except jwt.PyJWTError as e:
            print(f"PyJWT hatası: {e}")
        
        # Python-jose ile kontrol - audience parametresi eklendi
        try:
            payload = jose_jwt.decode(
                token, 
                jwt_secret, 
                algorithms=["HS256"],
                audience="authenticated"  # 'aud' parametresi token içeriğinden alındı
            )
            print("Jose JWT ile token doğrulandı:", payload)
            return True, payload
        except JWTError as e:
            print(f"Jose JWT hatası: {e}")
        
        # Alternatif olarak doğrulama yapmadan çözümleme
        try:
            # Token'ı doğrulama yapmadan çözümle
            header = jwt.get_unverified_header(token)
            payload = jwt.decode(token, options={"verify_signature": False})
            print("Doğrulanmamış token içeriği:")
            print("Header:", header)
            print("Payload:", payload)
            
            # Token'ı decode etmeyi başardık, ancak doğrulayamadık
            print(f"Token imzalama anahtarı: {jwt_secret[:10]}...")
            print(f"Token issuer: {payload.get('iss')}")
            print(f"Token audience: {payload.get('aud')}")
        except Exception as e:
            print(f"Token çözümlenemedi: {e}")
        
        return False, None
    except Exception as e:
        print(f"Beklenmeyen hata: {e}")
        return False, None

# Test kodu
if __name__ == "__main__":
    # Mevcut token alma kodu
    supabase: Client = create_client(url, key)
    response = supabase.auth.sign_in_with_password({
        "email": "muhammet.erengur@hotmail.com",
        "password": "MuhammetGur61."
    })
    token = response.session.access_token
    print("Alınan token:", token)
    
    # Token doğrulama
    is_valid, payload = verify_token(token)
    if is_valid:
        print("Token geçerli")
    else:
        print("Token geçersiz") 