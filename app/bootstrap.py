from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

import httpx
import uvicorn


@dataclass
class StartedMock:
    server: uvicorn.Server
    thread: threading.Thread


async def is_server_alive(base_url: str, timeout_s: float = 0.8) -> bool:
    """
    Prefer /health if available. If not, fall back to a cheap POST /chat probe.
    """
    base_url = base_url.rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        # 1) health endpoint
        try:
            r = await client.get("/health")
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            return False

        # 2) fallback probe (contract endpoint)
        try:
            r = await client.post("/chat", json={"user_id": "probe", "message": "hi"})
            return r.status_code < 500  # accept 2xx/4xx as "alive"
        except httpx.RequestError:
            return False


def _run_uvicorn(server: uvicorn.Server) -> None:
    # runs until server.should_exit = True
    server.run()


async def ensure_server_or_start_mock(base_url: str) -> StartedMock | None:
    """
    If base_url isn't reachable, start the mock FastAPI server on that host/port.
    Returns StartedMock if we started it; otherwise None.
    """
    alive = await is_server_alive(base_url)
    if alive:
        return None

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    # Import here to avoid dependency/side-effects if not needed
    from app.client.mock_server_app import app as mock_app

    config = uvicorn.Config(
        mock_app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)

    thread = threading.Thread(target=_run_uvicorn, args=(server,), daemon=True)
    thread.start()

    # Wait until it responds
    for _ in range(40):
        if await is_server_alive(base_url, timeout_s=0.5):
            return StartedMock(server=server, thread=thread)
        await asyncio.sleep(0.1)

    # If we got here, startup failed
    server.should_exit = True
    return None


def stop_mock(started: StartedMock | None) -> None:
    if not started:
        return
    started.server.should_exit = True
    # Best-effort join
    started.thread.join(timeout=2.0)