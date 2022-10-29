from typing import Union, Optional, Any, Sequence, MutableMapping

from asgi_lifespan import LifespanManager
from requests import PreparedRequest, Session, Response
from requests_toolbelt import sessions
import websockets.client

import asyncio
import json
import requests.adapters
from urllib.parse import unquote, urljoin, urlsplit
from httpx import AsyncClient

from starlette.testclient import (
    TestClient,
    ASGI3App,
)
from starlette.types import Scope, Message
from starlette.websockets import WebSocketDisconnect


class AsyncioWebSocketTestSession:
    def __init__(
        self,
        app: ASGI3App,
        scope: Scope,
        event_loop: asyncio.AbstractEventLoop,
        receive_queue: asyncio.Queue,
        send_queue: asyncio.Queue,
    ) -> None:
        self.event_loop = event_loop
        self.app = app
        self.scope = scope
        self.accepted_subprotocol = None
        self._receive_queue = receive_queue
        self._send_queue = send_queue

    async def __aenter__(self) -> "AsyncioWebSocketTestSession":
        self.event_loop.create_task(self._run())
        await self.send({"type": "websocket.connect"})
        message = await self.receive()
        assert message['type'] == 'websocket.accept'

        self.accepted_subprotocol = message.get("subprotocol", None)
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close(1000)

        while not self._send_queue.empty():
            message = await self._send_queue.get()
            if isinstance(message, BaseException):
                raise message

    async def _run(self) -> None:
        """
        The sub-thread in which the websocket session runs.
        """
        scope = self.scope
        receive = self._asgi_receive
        send = self._asgi_send
        try:
            await self.app(scope, receive, send)
        except BaseException as exc:
            await self._send_queue.put(exc)
            raise

    async def _asgi_receive(self) -> Message:
        while self._receive_queue.empty():
            await asyncio.sleep(0)
        return await self._receive_queue.get()

    async def _asgi_send(self, message: Message) -> None:
        await self._send_queue.put(message)

    def _raise_on_close(self, message: Message) -> None:
        if message["type"] == "websocket.close":
            raise WebSocketDisconnect(message.get("code", 1000))

    async def send(self, message: Message) -> None:
        await self._receive_queue.put(message)

    async def send_text(self, data: str) -> None:
        await self.send({"type": "websocket.receive", "text": data})

    async def send_bytes(self, data: bytes) -> None:
        await self.send({"type": "websocket.receive", "bytes": data})

    async def send_json(self, data: Any, mode: str = "text") -> None:
        assert mode in ["text", "binary"]
        text = json.dumps(data)
        if mode == "text":
            return await self.send({"type": "websocket.receive", "text": text})

        return await self.send({"type": "websocket.receive", "bytes": text.encode("utf-8")})

    async def close(self, code: int = 1000) -> None:
        await self.send({"type": "websocket.disconnect", "code": code})

    async def receive(self) -> Message:
        while True:
            try:
                message = self._send_queue.get_nowait()

                if isinstance(message, BaseException):
                    raise message

                return message
            except asyncio.queues.QueueEmpty:
                await asyncio.sleep(0.1)

    async def recv(self) -> str:
        message = await self.receive()
        self._raise_on_close(message)
        return message["text"]

    async def receive_bytes(self) -> bytes:
        message = await self.receive()
        self._raise_on_close(message)
        return message["bytes"]

    async def receive_json(self, mode: str = "text") -> Any:
        assert mode in ["text", "binary"]
        message = await self.receive()
        self._raise_on_close(message)

        if mode == "text":
            text = message["text"]
        else:
            text = message["bytes"].decode("utf-8")

        return json.loads(text)


class _Upgrade(Exception):
    def __init__(self, session: AsyncioWebSocketTestSession) -> None:
        self.session = session


class _AsyncioASGIAdapter(requests.adapters.HTTPAdapter):
    def __init__(
        self,
        app: ASGI3App,
        event_loop: asyncio.AbstractEventLoop,
        receive_queue: asyncio.Queue,
        send_queue: asyncio.Queue,
        raise_server_exceptions: bool = True,
        root_path: str = "",
    ) -> None:
        super().__init__()
        self.event_loop = event_loop
        self.app = app
        self.raise_server_exceptions = raise_server_exceptions
        self.root_path = root_path
        self.receive_queue = receive_queue
        self.send_queue = send_queue

    def send(
        self,
        request: PreparedRequest,
        *args: Any,
        **kwargs: Any
    ) -> None:
        scheme, netloc, path, query, fragment = (
            str(item) for item in urlsplit(request.url)
        )

        default_port = {"http": 80, "ws": 80, "https": 443, "wss": 443}[scheme]

        if ":" in netloc:
            host, port_string = netloc.split(":", 1)
            port = int(port_string)
        else:
            host = netloc
            port = default_port

        # Include the 'host' header.
        if "host" in request.headers:
            headers: list[tuple[bytes, bytes]] = []
        elif port == default_port:
            headers = [(b"host", host.encode())]
        else:
            headers = [(b"host", (f"{host}:{port}").encode())]

        # Include other request headers.
        headers += [
            (key.lower().encode(), value.encode())
            for key, value in request.headers.items()
        ]

        if scheme not in {"ws", "wss"}:
            raise ValueError('Available only for websockets connection')

        subprotocol = request.headers.get("sec-websocket-protocol", None)

        if subprotocol is None:
            subprotocols: Sequence[str] = []
        else:
            subprotocols = [value.strip() for value in subprotocol.split(",")]

        scope = {
            "type": "websocket",
            "path": unquote(path),
            "root_path": self.root_path,
            "scheme": scheme,
            "query_string": query.encode(),
            "headers": headers,
            "client": ["testclient", 50000],
            "server": [host, port],
            "subprotocols": subprotocols,
        }
        session = AsyncioWebSocketTestSession(
            self.app,
            scope,
            self.event_loop,
            receive_queue=self.receive_queue,
            send_queue=self.send_queue

        )
        raise _Upgrade(session)


class AsyncioTestClient(Session):
    def __init__(
        self,
        app: ASGI3App,
        base_url: str = "http://testserver",
        raise_server_exceptions: bool = True,
        root_path: str = "",
        event_loop: Optional[asyncio.AbstractEventLoop] = None
    ) -> None:
        super().__init__()

        self.receive_queue = asyncio.Queue()
        self.send_queue = asyncio.Queue()
        self.event_loop = event_loop or asyncio.get_event_loop()

        adapter = _AsyncioASGIAdapter(
            app,
            event_loop=self.event_loop,
            receive_queue=self.receive_queue,
            send_queue=self.send_queue,
            raise_server_exceptions=raise_server_exceptions,
            root_path=root_path,
        )
        self.mount("http://", adapter)
        self.mount("https://", adapter)
        self.mount("ws://", adapter)
        self.mount("wss://", adapter)
        self.headers.update({"user-agent": "testclient"})
        self.app = app
        self.base_url = base_url

    def websocket_connect(
        self,
        url: str,
        subprotocols: Optional[Sequence[str]] = None,
        **kwargs: Any
    ) -> Any:
        url = urljoin("ws://testserver", url)

        headers = kwargs.get("headers", {})
        headers.setdefault("connection", "upgrade")
        headers.setdefault("sec-websocket-key", "testserver==")
        headers.setdefault("sec-websocket-version", "13")

        if subprotocols is not None:
            headers.setdefault("sec-websocket-protocol", ", ".join(subprotocols))

        kwargs["headers"] = headers

        try:
            super().request("GET", url, **kwargs)
        except _Upgrade as exc:
            session = exc.session
        else:
            raise RuntimeError("Expected WebSocket upgrade")  # pragma: no cover

        return session


class AppClient:
    def __init__(
        self,
        client: AsyncClient,
    ):
        self.client = client

        self.headers = {}
        self.ws_stem = f'/events'

    async def set_public_token(self):
        response = await self.client.post('/token/all')
        token = response.json()['access_token']
        self.set_token(token)

    def set_token(self, token: str):
        self.headers = {
            "Authorization": f"Bearer {token}",
        }
        self.ws_stem = f'/events?token={token}'

    async def get(self, *args, **kwargs):
        return await self.client.get(*args, **kwargs, headers=self.headers)

    async def post(self, *args, **kwargs):
        return await self.client.post(*args, **kwargs, headers=self.headers)

    async def delete(self, *args, **kwargs):
        return await self.client.delete(*args, **kwargs, headers=self.headers)

    async def __aenter__(self, *args, **kwargs):
        await self.client.__aenter__(*args, **kwargs)
        return self

    async def __aexit__(self, *args, **kwargs):
        await self.client.__aexit__(*args, **kwargs)

    def websocket_connect(self):
        raise NotImplementedError


class LocalAppClient(AppClient):
    def __init__(self, client: AsyncClient,
                 ws_client: AsyncioTestClient,
                 lifespan_manager: LifespanManager):
        super().__init__(client)
        self.ws_client = ws_client
        self.lifespan = lifespan_manager

    def websocket_connect(self):
        return self.ws_client.websocket_connect(self.ws_stem)

    async def __aenter__(self, *args, **kwargs):
        await self.lifespan.__aenter__(*args, **kwargs)
        return await super().__aenter__(*args, **kwargs)

    async def __aexit__(self, *args, **kwargs):
        await super().__aexit__(*args, **kwargs)
        await self.lifespan.__aexit__(*args, **kwargs)


class RemoteAppClient(AppClient):
    def websocket_connect(self):
        ws_base_url = str(self.client.base_url).replace('http', 'ws')
        ws_url = f'{ws_base_url}{self.ws_stem}'
        return websockets.client.connect(ws_url)
