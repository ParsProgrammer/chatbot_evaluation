import httpx


async def is_alive(base_url: str, timeout_s: float = 0.8) -> bool:
    """
    Prefer GET /health. If not present, probe POST /chat.
    """
    base_url = base_url.rstrip("/")
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s) as client:
        try:
            r = await client.get("/health")
            if r.status_code == 200:
                return True
        except httpx.RequestError:
            return False

        try:
            r = await client.post("/chat", json={"user_id": "probe", "message": "hi"})
            return r.status_code < 500  # 2xx/4xx = alive
        except httpx.RequestError:
            return False