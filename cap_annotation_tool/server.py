from __future__ import annotations

import json
from datetime import datetime, timezone
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parent
ANNOTATIONS_PATH = ROOT / "annotations.json"
TASKS_PATH = ROOT / "tasks.json"


class AnnotationHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def _send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)
        if path == "/api/tasks":
            if not TASKS_PATH.exists():
                self._send_json({"error": "tasks.json not found"}, 404)
                return
            self._send_json(json.loads(TASKS_PATH.read_text(encoding="utf-8")))
            return
        if path == "/api/annotations":
            if not ANNOTATIONS_PATH.exists():
                self._send_json({"version": 1, "frames": {}})
                return
            self._send_json(json.loads(ANNOTATIONS_PATH.read_text(encoding="utf-8")))
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path != "/api/annotations":
            self._send_json({"error": "unknown endpoint"}, 404)
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:
            self._send_json({"error": f"invalid json: {exc}"}, 400)
            return
        payload["saved_at"] = datetime.now(timezone.utc).isoformat()
        ANNOTATIONS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        self._send_json({"ok": True, "path": str(ANNOTATIONS_PATH)})


def main() -> None:
    host = "127.0.0.1"
    port = 8765
    server = ThreadingHTTPServer((host, port), AnnotationHandler)
    print(f"Annotation tool: http://{host}:{port}/")
    print(f"Saving annotations to: {ANNOTATIONS_PATH}")
    server.serve_forever()


if __name__ == "__main__":
    main()
