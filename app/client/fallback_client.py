import asyncio
from typing import Optional, Tuple, Dict, Any

from app.client.base import ChatClient
from app.client.http_client import HttpChatClient
from app.client.mock_server import start_mock_if_needed, stop_mock, StartedMock
from app.client.health import is_alive


class FallbackChatClient(ChatClient):
    """
    - If base_url is alive: use it
    - Else: start local FastAPI mock on same host/port and use it
    """

    def __init__(self, base_url: str, timeout_s: float = 10.0, verbose: bool = False):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.verbose = verbose

        self._http: Optional[HttpChatClient] = None
        self._started_mock: Optional[StartedMock] = None
        self._ready = False

        # ✅ prevents concurrent mock-start attempts
        self._ready_lock = asyncio.Lock()

    async def _ensure_ready(self) -> None:
        if self._ready:
            return

        async with self._ready_lock:
            # double-check after acquiring lock
            if self._ready:
                return

            alive = await is_alive(self.base_url)
            if alive:
                if self.verbose:
                    print(f"[info] Chatbot reachable at {self.base_url}. Using real server.")
                self._started_mock = None
            else:
                self._started_mock = await start_mock_if_needed(self.base_url)
                if self._started_mock:
                    if self.verbose:
                        print(f"[info] No chatbot reachable at {self.base_url}. Started local mock server.")
                else:
                    raise RuntimeError(
                        f"Chatbot not reachable at {self.base_url}, and failed to start local mock server."
                    )

            self._http = HttpChatClient(self.base_url, timeout_s=self.timeout_s)
            self._ready = True

    async def chat(self, user_id: str, message: str) -> Tuple[Dict[str, Any], Optional[float]]:
        await self._ensure_ready()
        assert self._http is not None
        return await self._http.chat(user_id=user_id, message=message)

    async def aclose(self) -> None:
        if self._http is not None:
            await self._http.aclose()
        stop_mock(self._started_mock)