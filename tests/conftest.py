"""Test bootstrap: make src/ importable without requiring an editable install."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _fresh_ollama_guard():
    """Each test starts with no cached "this model runs locally" answers."""
    from alfred.core import ollama_guard

    ollama_guard.reset()
    yield
    ollama_guard.reset()


class FakeOllama:
    """A real HTTP server on 127.0.0.1 that answers like Ollama.

    Records every request as (path, json body). The path is whatever the
    client sent on the request line, so a request that came through it as a
    proxy shows up with an absolute URI ("http://127.0.0.1:N/api/chat").
    """

    def __init__(self) -> None:
        import json as _json
        import threading
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from urllib.parse import urlsplit

        self.requests: list[tuple[str, dict]] = []
        self.show_status = 200
        self.show_reply: dict = {"details": {"format": "gguf"}, "capabilities": ["completion"]}
        fake = self

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):  # keep test output quiet
                pass

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                raw = self.rfile.read(length) if length else b""
                try:
                    body = _json.loads(raw or b"{}")
                except ValueError:
                    body = {}
                fake.requests.append((self.path, body))
                route = urlsplit(self.path).path
                status, reply = 200, {}
                if route == "/api/show":
                    status, reply = fake.show_status, fake.show_reply
                elif route == "/api/chat":
                    reply = {"message": {"role": "assistant", "content": "ok"}}
                elif route == "/api/embeddings":
                    reply = {"embedding": [0.1, 0.2, 0.3]}
                else:
                    status, reply = 404, {"error": "not found"}
                data = _json.dumps(reply).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"
        self._thread = threading.Thread(
            target=self._server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True,
        )
        self._thread.start()

    @property
    def paths(self) -> list[str]:
        return [path for path, _ in self.requests]

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()


@pytest.fixture
def fake_ollama():
    """Factory: each call starts another FakeOllama; all are stopped after the test."""
    servers: list[FakeOllama] = []

    def _start() -> FakeOllama:
        server = FakeOllama()
        servers.append(server)
        return server

    yield _start
    for server in servers:
        server.close()
