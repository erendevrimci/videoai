import os
import redis.asyncio as redis
from dotenv import load_dotenv
import asyncio
import threading

load_dotenv()

# Redis URL'sini ortam değişkeninden al
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Asenkron Redis bağlantı havuzu oluştur
# Bu, birden çok isteğin aynı anda verimli bir şekilde bağlantı kullanmasını sağlar.
redis_pool = redis.ConnectionPool.from_url(REDIS_URL, decode_responses=True)

async def publish_message(channel: str, message: str):
    """Belirtilen kanala bir mesaj yayınlar."""
    async with redis.Redis(connection_pool=redis_pool) as r:
        await r.publish(channel, message)

async def subscribe_to_channel(channel: str):
    """
    Belirtilen kanala abone olur ve mesajları dinler.
    Bu bir 'async generator' fonksiyonudur, yani her gelen mesajda bir değer 'yield' eder.
    """
    async with redis.Redis(connection_pool=redis_pool) as r:
        pubsub = r.pubsub()
        await pubsub.subscribe(channel)
        
        # Abone olunan kanaldan sürekli olarak mesajları dinle
        async for message in pubsub.listen():
            # Sadece 'message' türündeki mesajları işle
            if message["type"] == "message":
                yield message["data"]

# --- Senkron Ortamdan Asenkron Çağrı Yardımcıları ---

def run_async(coro):
    """
    Verilen coroutine'i ayrı bir thread'de yeni bir event loop üzerinde çalıştırır.
    Bu, senkron bir fonksiyondan (örn. Celery task) asenkron bir fonksiyonu
    güvenli bir şekilde çağırmak için kullanılır.
    """
    async def main_wrapper():
        await coro
        # Redis bağlantısının düzgün kapanması gibi arka plan görevlerine
        # zaman tanımak için bir anlık bekleme ekliyoruz. Bu, "Event loop is closed"
        # hatasını önler.
        await asyncio.sleep(0)

    def run():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(main_wrapper())
        finally:
            # Artık döngüyü kapatmak güvenli.
            loop.close()

    thread = threading.Thread(target=run)
    thread.start()
    # Ana thread'i bloklamamak için join() yapmıyoruz.

def publish_sync(channel: str, message: str):
    """
    Senkron bir bağlamdan mesaj yayınlamak için kullanılır.
    Asenkron `publish_message` fonksiyonunu arka planda çalıştırır.
    """
    run_async(publish_message(channel, message)) 