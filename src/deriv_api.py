"""
Deriv WebSocket client helper for market data and trading requests.
"""

import asyncio
import json
import logging
from typing import Any, Dict, Optional

import websockets

logger = logging.getLogger(__name__)


class DerivWebSocket:
    """Minimal Deriv WebSocket client with request/response handling."""

    def __init__(self, *, app_id: str, token: str = "", ws_url: str = "wss://ws.deriv.com/websockets/v3"):
        self.app_id = str(app_id)
        self.token = token or ""
        self.ws_url = ws_url.rstrip("/")
        self._ws = None
        self._req_id = 1
        self._pending: Dict[int, asyncio.Future] = {}
        self._queue: asyncio.Queue = asyncio.Queue()
        self._listener_task: Optional[asyncio.Task] = None

    def _endpoint(self) -> str:
        if "?" in self.ws_url:
            return self.ws_url
        return f"{self.ws_url}?app_id={self.app_id}"

    async def connect(self) -> None:
        if self._ws is not None:
            return
        last_error = None
        endpoints = [self._endpoint()]
        if "derivws.com" not in self.ws_url:
            fallback = "wss://ws.derivws.com/websockets/v3"
            endpoints.append(f"{fallback}?app_id={self.app_id}")

        for attempt in range(5):
            for endpoint in endpoints:
                try:
                    self._ws = await websockets.connect(
                        endpoint,
                        ping_interval=20,
                        ping_timeout=20,
                        open_timeout=10,
                        close_timeout=5,
                    )
                    self.ws_url = endpoint.split("?")[0]
                    self._listener_task = asyncio.create_task(self._listen())
                    return
                except Exception as exc:
                    last_error = exc
                    await asyncio.sleep(1.0 + attempt)
                    continue

        raise last_error

    async def close(self) -> None:
        if self._listener_task:
            self._listener_task.cancel()
        if self._ws:
            await self._ws.close()
        self._ws = None

    async def authorize(self) -> Dict[str, Any]:
        if not self.token:
            raise RuntimeError("Deriv token is required for authorization")
        return await self.request({"authorize": self.token})

    async def request(self, payload: Dict[str, Any], timeout: float = 10.0) -> Dict[str, Any]:
        if self._ws is None:
            raise RuntimeError("WebSocket not connected")
        req_id = self._req_id
        self._req_id += 1
        payload["req_id"] = req_id
        fut = asyncio.get_running_loop().create_future()
        self._pending[req_id] = fut
        await self._ws.send(json.dumps(payload))
        return await asyncio.wait_for(fut, timeout=timeout)

    async def next_message(self) -> Dict[str, Any]:
        return await self._queue.get()

    async def _listen(self) -> None:
        try:
            while True:
                raw = await self._ws.recv()
                msg = json.loads(raw)
                req_id = msg.get("req_id")
                if req_id in self._pending:
                    fut = self._pending.pop(req_id)
                    if not fut.cancelled():
                        fut.set_result(msg)
                    continue
                await self._queue.put(msg)
        except asyncio.CancelledError:
            return
        except Exception as exc:
            logger.warning(f"Deriv WebSocket listener error: {exc}")
            try:
                self._queue.put_nowait({"msg_type": "_ws_error", "error": str(exc)})
            except Exception:
                pass
            # Flush pending requests with errors
            for req_id, fut in list(self._pending.items()):
                if not fut.cancelled():
                    fut.set_exception(exc)
            self._pending.clear()
