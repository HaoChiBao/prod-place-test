"""Serve the catalog UI and product swaps.

    python ui_server.py

Pick an object, name a product, and the swap runs in the background. The page
polls until the result can play beside the original window.
"""

from __future__ import annotations

import json
import re
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from store import get_clip, list_clips, load_dotenv
from swap import SwapBusy, SwapDeps, decode_image, get_swap, list_swaps, start_swap, swap_path

INDEX = Path(__file__).with_name("ui") / "index.html"
_ID = re.compile(r"^[0-9a-f]{6,32}$")
_SWAP_POST = re.compile(r"^/api/clips/([0-9a-f]{6,32})/objects/([0-9a-f]{6,32})/swap$")
_SWAP_LIST = re.compile(r"^/api/clips/([0-9a-f]{6,32})/swaps$")
_SWAP_GET = re.compile(r"^/api/swaps/([0-9a-f]{6,32})$")
_SWAP_FILE = re.compile(r"^/swaps/([0-9a-f]{6,32})/(window|result)$")
_TYPES = {
    ".mp4": "video/mp4",
    ".webm": "video/webm",
    ".mov": "video/quicktime",
}


class Handler(BaseHTTPRequestHandler):
    db_root: Path | None = None
    deps: SwapDeps | None = None

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
        listed = _SWAP_LIST.fullmatch(path)
        if listed:
            self._json(list_swaps(listed.group(1)))
            return
        if path.startswith("/api/clips/"):
            clip_id = path.removeprefix("/api/clips/")
            clip = get_clip(clip_id, self.db_root) if _ID.fullmatch(clip_id) else None
            if clip is None:
                self._json({"error": "not found"}, status=404)
                return
            self._json(clip)
            return
        found = _SWAP_GET.fullmatch(path)
        if found:
            job = get_swap(found.group(1))
            if job is None:
                self._json({"error": "not found"}, status=404)
                return
            self._json(job)
            return
        media = _SWAP_FILE.fullmatch(path)
        if media:
            file = swap_path(self.db_root, media.group(1), media.group(2))
            if file is None:
                self._json({"error": "not found"}, status=404)
                return
            self._send_file(file, "video/mp4")
            return
        if path.startswith("/media/"):
            self._media(path.removeprefix("/media/"))
            return
        self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:
        match = _SWAP_POST.fullmatch(urlparse(self.path).path)
        if not match:
            self._json({"error": "not found"}, status=404)
            return
        clip_id, object_id = match.group(1), match.group(2)
        try:
            payload = self._json_body()
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return
        clip = get_clip(clip_id, self.db_root)
        if clip is None:
            self._json({"error": "not found"}, status=404)
            return
        obj = next((item for item in clip["objects"] if item["id"] == object_id), None)
        if obj is None:
            self._json({"error": "not found"}, status=404)
            return
        product = payload.get("product")
        if not isinstance(product, str) or not product.strip() or len(product.strip()) > 200:
            self._json({"error": "enter a product name"}, status=400)
            return
        try:
            image = decode_image(payload)
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
            return
        try:
            job = start_swap(
                clip_id=clip_id,
                object_id=object_id,
                product=product.strip(),
                image=image,
                root=self.db_root,
                deps=self.deps,
            )
        except SwapBusy as exc:
            self._json({"error": str(exc)}, status=409)
            return
        self._json(job, status=202)

    def _json_body(self) -> dict:
        raw_len = self.headers.get("Content-Length")
        if raw_len is None:
            raise ValueError("expected JSON")
        try:
            length = int(raw_len)
        except ValueError:
            raise ValueError("expected JSON") from None
        if length < 2 or length > 12_000_000:
            raise ValueError("image is too large" if length > 12_000_000 else "expected JSON")
        try:
            data = json.loads(self.rfile.read(length).decode())
        except (UnicodeDecodeError, json.JSONDecodeError):
            raise ValueError("expected JSON") from None
        if not isinstance(data, dict):
            raise ValueError("expected JSON")
        return data

    def _media(self, clip_id: str) -> None:
        clip = get_clip(clip_id, self.db_root) if _ID.fullmatch(clip_id) else None
        if clip is None:
            self._json({"error": "not found"}, status=404)
            return
        file = Path(clip["video_path"])
        if not file.is_file():
            self._json({"error": "video missing"}, status=404)
            return
        content_type = _TYPES.get(file.suffix.lower(), "application/octet-stream")
        self._send_file(file, content_type)

    def _send_file(self, file: Path, content_type: str) -> None:
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


def serve(
    root: Path | None = None,
    host: str = "127.0.0.1",
    port: int = 8765,
    deps: SwapDeps | None = None,
) -> ThreadingHTTPServer:
    handler = type("BoundHandler", (Handler,), {"db_root": root, "deps": deps})
    return ThreadingHTTPServer((host, port), handler)


def main(argv: list[str] | None = None) -> int:
    load_dotenv()
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
