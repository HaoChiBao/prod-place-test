"""Serve the catalog UI: one list of objects per clip, click to seek and draw the box.

    python ui_server.py
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from store import get_clip, list_clips

INDEX = Path(__file__).with_name("ui") / "index.html"
_ID = re.compile(r"^[0-9a-f]{6,32}$")
_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


class Handler(BaseHTTPRequestHandler):
    db_root: Path | None = None

    def log_message(self, fmt: str, *args) -> None:
        return

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._bytes(INDEX.read_bytes(), "text/html; charset=utf-8")
            return
        if path == "/api/clips":
            self._json(list_clips(self.db_root))
            return
        if path.startswith("/api/clips/"):
            clip_id = path.removeprefix("/api/clips/")
            clip = get_clip(clip_id, self.db_root) if _ID.match(clip_id) else None
            if clip is None:
                self._json({"error": "not found"}, status=404)
                return
            self._json(clip)
            return
        if path.startswith("/media/"):
            self._media(path.removeprefix("/media/"))
            return
        self._json({"error": "not found"}, status=404)

    def _media(self, clip_id: str) -> None:
        clip = get_clip(clip_id, self.db_root) if _ID.match(clip_id) else None
        if clip is None:
            self._json({"error": "not found"}, status=404)
            return
        file = Path(clip["video_path"])
        if not file.is_file():
            self._json({"error": "video missing"}, status=404)
            return
        content_type = _TYPES.get(file.suffix.lower(), "application/octet-stream")
        size = file.stat().st_size
        range_header = self.headers.get("Range")
        start, end = 0, size - 1
        status = 200
        if range_header and range_header.startswith("bytes="):
            spec = range_header.split("=", 1)[1].split(",")[0].strip()
            left, _, right = spec.partition("-")
            try:
                start = int(left) if left else 0
                end = int(right) if right else size - 1
            except ValueError:
                start, end = 0, size - 1
            else:
                end = min(end, size - 1)
                if start < 0 or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{size}")
                    self.end_headers()
                    return
                status = 206
        length = end - start + 1
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(length))
        if status == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with file.open("rb") as handle:
            handle.seek(start)
            remaining = length
            while remaining:
                chunk = handle.read(min(65536, remaining))
                if not chunk:
                    break
                self.wfile.write(chunk)
                remaining -= len(chunk)

    def _json(self, payload, status: int = 200) -> None:
        body = json.dumps(payload).encode()
        self._bytes(body, "application/json", status)

    def _bytes(self, body: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)


def serve(root: Path | None = None, host: str = "127.0.0.1", port: int = 8765) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"db_root": root})
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    port = 8765
    if len(args) == 2 and args[0] == "--port":
        port = int(args[1])
    elif args:
        print("usage: python ui_server.py [--port 8765]", file=sys.stderr)
        return 2
    httpd = serve(port=port)
    print(f"http://{httpd.server_address[0]}:{httpd.server_address[1]}", flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
