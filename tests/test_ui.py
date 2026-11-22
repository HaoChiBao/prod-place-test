"""Catalog API: each clip lists its objects, and the video can be fetched."""

from __future__ import annotations

import json
import threading
import urllib.request
from pathlib import Path

from store import save_findings
from ui_server import serve


def _objects():
    return [
        {
            "label": "green wine bottle",
            "category": "bottle",
            "suggested_product": "branded wine bottle",
            "start_s": 2.0,
            "end_s": 4.0,
            "visibility": 0.9,
            "boxes": [{"t": 2.15, "x": 80, "y": 40, "w": 30, "h": 80}],
        },
        {
            "label": "wine glass",
            "category": "glass",
            "suggested_product": "branded glass",
            "start_s": 0.0,
            "end_s": 2.0,
            "visibility": 0.7,
            "boxes": [{"t": 0.15, "x": 10, "y": 10, "w": 20, "h": 40}],
        },
    ]


def test_ui_lists_objects_and_serves_the_video(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42fake-video-bytes")
    save_findings(video, _objects(), 0.025, root=tmp_path)
    other = tmp_path / "other.mp4"
    other.write_bytes(b"second")
    save_findings(
        other,
        [
            {
                "label": "coffee mug",
                "category": "cup",
                "suggested_product": "branded mug",
                "start_s": 1.0,
                "end_s": 1.0,
                "visibility": 0.5,
                "boxes": [{"t": 1.0, "x": 1, "y": 2, "w": 3, "h": 4}],
            }
        ],
        None,
        root=tmp_path,
    )
    httpd = serve(tmp_path, port=0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{httpd.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "currentTime" in page
        assert 'id="box"' in page
        clips = json.load(urllib.request.urlopen(base + "/api/clips"))
        assert [clip["name"] for clip in clips] == ["clip.mp4", "other.mp4"]
        bottle = next(obj for obj in clips[0]["objects"] if obj["label"] == "green wine bottle")
        assert (bottle["start_s"], bottle["end_s"]) == (2.0, 4.0)
        assert bottle["boxes"][0]["x"] == 80
        body = urllib.request.urlopen(base + "/media/" + clips[0]["id"]).read()
        assert body.startswith(b"\x00\x00\x00\x18ftypmp42")
        req = urllib.request.Request(base + "/media/" + clips[0]["id"], headers={"Range": "bytes=0-3"})
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            assert resp.read() == b"\x00\x00\x00\x18"
    finally:
        httpd.shutdown()
