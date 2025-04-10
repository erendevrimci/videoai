from supabase import create_client, Client
from dotenv import load_dotenv
import os

load_dotenv()
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase: Client = create_client(url, key)

# Kullanıcı girişi
response = supabase.auth.sign_in_with_password({
    "email": "muhammet.erengur@hotmail.com",
    "password": "MuhammetGur61."
})

# JWT token'i al
token = response.session.access_token
print("JWT Token:", token)