from fastapi import WebSocket
from typing import Dict

class ConnectionManager:
    def __init__(self):
        # Aktif bağlantıları saklamak için bir sözlük kullanıyoruz.
        # Anahtar: client_id (örn: task_id), Değer: WebSocket nesnesi
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, client_id: str, websocket: WebSocket):
        """Yeni bir WebSocket bağlantısını kabul eder ve saklar."""
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        """Bir WebSocket bağlantısını listeden kaldırır."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_personal_message(self, message: str, client_id: str):
        """Belirli bir istemciye mesaj gönderir."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_text(message)

    async def send_json_message(self, data: dict, client_id: str):
        """Belirli bir istemciye JSON formatında mesaj gönderir."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(data)

# Tek bir global manager nesnesi oluşturuyoruz, böylece uygulama boyunca aynı nesne kullanılır.
manager = ConnectionManager() 