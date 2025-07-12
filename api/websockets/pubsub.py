import os
import redis.asyncio as redis
from dotenv import load_dotenv

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