import time
import httpx
from typing import Optional, Tuple, Dict, Any

from app.client.base import ChatClient


class HttpChatClient(ChatClient):
    def __init__(self, base_url: str, timeout_s: float = 10.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s),
            headers={"Content-Type": "application/json"},
        )

    async def chat(self, user_id: str, message: str) -> Tuple[Dict[str, Any], Optional[float]]:
        payload = {"user_id": user_id, "message": message}
        t0 = time.perf_counter()
        resp = await self._client.post("/chat", json=payload)
        latency_ms = (time.perf_counter() - t0) * 1000.0
        resp.raise_for_status()
        return resp.json(), float(latency_ms)

    async def aclose(self) -> None:
        await self._client.aclose()