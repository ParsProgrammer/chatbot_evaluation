import asyncio
import threading
from dataclasses import dataclass
from urllib.parse import urlparse

import uvicorn

from app.client.health import is_alive


@dataclass
class StartedMock:
    server: uvicorn.Server
    thread: threading.Thread


def _run(server: uvicorn.Server) -> None:
    server.run()


async def start_mock_if_needed(base_url: str) -> StartedMock | None:
    if await is_alive(base_url):
        return None

    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or (443 if parsed.scheme == "https" else 80)

    from app.client.mock_server_app import app as mock_app

    config = uvicorn.Config(
        mock_app,
        host=host,
        port=port,
        log_level="warning",
        access_log=False,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=_run, args=(server,), daemon=True)
    thread.start()

    # Wait until it responds
    for _ in range(60):
        if await is_alive(base_url, timeout_s=0.5):
            return StartedMock(server=server, thread=thread)
        await asyncio.sleep(0.1)

    server.should_exit = True
    return None


def stop_mock(started: StartedMock | None) -> None:
    if not started:
        return
    started.server.should_exit = True
    started.thread.join(timeout=2.0)