from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any


class ChatClient(ABC):
    @abstractmethod
    async def chat(self, user_id: str, message: str) -> Tuple[Dict[str, Any], Optional[float]]:
        """Returns (response_json, measured_latency_ms)."""
        raise NotImplementedError

    @abstractmethod
    async def aclose(self) -> None:
        raise NotImplementedError