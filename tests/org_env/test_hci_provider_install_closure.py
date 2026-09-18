"""The focused HCI install must serve the real SDK route, without paid calls."""

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from environments.org_env.llm import client as client_module
from environments.org_env.llm.client import OpenAIOrgLLMClient


def test_openai_compatible_sdk_and_watchdog_reach_local_http(monkeypatch):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            size = int(self.headers["Content-Length"])
            requests.append((self.path, json.loads(self.rfile.read(size))))
            content = ("{\"approved\": true}" if requests[-1][1].get("response_format")
                       else "local-ok")
            payload = json.dumps({
                "id": "chatcmpl-local", "object": "chat.completion", "created": 1,
                "model": "local-model", "choices": [{"index": 0, "finish_reason": "stop",
                    "message": {"role": "assistant", "content": content}}],
                "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7},
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    # The direct path still uses the production SDK and watchdog. The child
    # process deadline used on Windows/worker threads has separate tests.
    monkeypatch.setattr(client_module, "_requires_provider_process_deadline", lambda: False)
    try:
        client = OpenAIOrgLLMClient(model="local-model", api_key="local-only",
            base_url=f"http://127.0.0.1:{server.server_port}/v1", max_retries=0)
        assert client.generate_text("system", "hello") == "local-ok"
        assert client.generate_json("system", "review", {"type": "object",
            "properties": {"approved": {"type": "boolean"}}, "required": ["approved"]}) == {"approved": True}
        assert len(requests) == 2
        assert all(path == "/v1/chat/completions" for path, _ in requests)
        assert requests[1][1]["response_format"] == {"type": "json_object"}
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
