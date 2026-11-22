"""Window sizing, the Wan request, polling, and muxing the original audio. HTTP is mocked."""

from __future__ import annotations

import base64
import io
import json
import subprocess
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from store import list_clips, save_findings
from swap import (
    API,
    HttpError,
    SwapBusy,
    SwapDeps,
    _public_error,
    aspects_for,
    choose_aspect,
    choose_duration,
    download_output,
    durations_for,
    finish_clip,
    finish_plan,
    generate_clip,
    get_swap,
    place_window,
    poll_budget_s,
    poll_video,
    position_phrase,
    start_swap,
    submit_video,
    reset_jobs,
    swap_prompt,
    urllib_transport,
    video_request,
)
from ui_server import serve

_KEY = "test-openrouter-key"
_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


@pytest.fixture(autouse=True)
def _clear_swaps():
    reset_jobs()
    yield
    reset_jobs()


def _clock():
    state = {"t": 0.0}

    def now():
        return state["t"]

    def sleep(seconds):
        state["t"] += seconds

    return state, now, sleep


def _ffmpeg(cmd: list[str]) -> None:
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _color(path: Path, color: str, size: str, seconds: int, fps: int, audio: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:d={seconds}:r={fps}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", f"sine=frequency=440:d={seconds}"]
    cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(path))
    _ffmpeg(cmd)


def _probe(path: Path) -> dict:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate:format=duration",
            "-of",
            "json",
            str(path),
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    data = json.loads(proc.stdout)
    video = next(stream for stream in data["streams"] if stream["codec_type"] == "video")
    num, _, den = str(video["avg_frame_rate"]).partition("/")
    fps = float(num) / float(den or 1)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "has_audio": any(stream["codec_type"] == "audio" for stream in data["streams"]),
        "duration": float(data["format"]["duration"]),
    }


def _pixel(path: Path, x: int, y: int) -> tuple[int, int, int]:
    info = _probe(path)
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    index = (y * info["width"] + x) * 3
    raw = proc.stdout
    return (raw[index], raw[index + 1], raw[index + 2])


def _rgb(path: Path) -> tuple[float, float, float]:
    proc = subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "pipe:1",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw = proc.stdout
    count = len(raw) // 3
    return (sum(raw[0::3]) / count, sum(raw[1::3]) / count, sum(raw[2::3]) / count)


def test_request_names_the_object_the_position_and_the_lighting():
    phrase = position_phrase({"x": 650, "y": 400, "w": 100, "h": 200}, 1000, 1000)
    assert phrase == "center-right of the frame"
    prompt = swap_prompt("green wine bottle", phrase, "Acme Cola can", reference_image=True)
    assert "green wine bottle" in prompt
    assert "center-right of the frame" in prompt
    assert "Acme Cola can" in prompt
    for clause in (
        "exactly the same position",
        "lighting direction",
        "color temperature",
        "contact shadows",
        "specular highlights",
        "depth of field",
        "motion blur",
    ):
        assert clause in prompt
    assert "locked" in prompt
    body = video_request(
        prompt,
        "https://media.example/window.mp4",
        4,
        99,
        image_url="https://media.example/can.png",
    )
    assert body == {
        "model": "alibaba/wan-3.0",
        "prompt": prompt,
        "input_references": [
            {"type": "video_url", "video_url": {"url": "https://media.example/window.mp4"}},
            {"type": "image_url", "image_url": {"url": "https://media.example/can.png"}},
        ],
        "duration": 4,
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "generate_audio": False,
        "seed": 99,
    }
    bare = video_request(prompt, "https://media.example/window.mp4", 2, 1)
    assert [ref["type"] for ref in bare["input_references"]] == ["video_url"]
    with pytest.raises(ValueError, match="https"):
        video_request(prompt, "data:video/mp4;base64,AAAA", 2, 1)


def test_aspect_follows_the_frame_instead_of_stretching_it():
    listed = ["16:9", "4:3", "1:1", "3:4", "9:16", "21:9"]
    payload = {"data": [{"id": "alibaba/wan-3.0", "supported_aspect_ratios": listed}]}
    assert aspects_for(payload) == listed
    assert choose_aspect(1920, 1080, listed) == "16:9"
    assert choose_aspect(1080, 1920, listed) == "9:16"
    assert choose_aspect(1080, 1080, listed) == "1:1"
    assert choose_aspect(1440, 1080, listed) == "4:3"
    assert choose_aspect(1080, 1440, listed) == "3:4"
    assert video_request("prompt", "https://media.example/v.mp4", 2, 1, aspect_ratio="9:16")["aspect_ratio"] == "9:16"


def test_window_is_padded_and_clamped_to_the_model_max():
    supported, model_max = durations_for(
        {
            "data": [
                {"id": "other/model", "supported_durations": [1]},
                {"id": "alibaba/wan-3.0", "supported_durations": [-1, "2", 4, 6, 8, 10, 30]},
            ]
        }
    )
    assert supported == [2, 4, 6, 8, 10, 30]
    assert model_max == 30
    capped = [d for d in supported if d <= 10]

    assert choose_duration(0.4, capped, 30, 10) == 2
    assert place_window(5.0, 5.4, 30, 2) == pytest.approx((4.2, 6.2), abs=0.02)

    assert choose_duration(25, capped, 40, 10) == 10
    start, end = place_window(0.0, 25.0, 40, 10)
    assert (start, end) == pytest.approx((7.5, 17.5), abs=0.05)
    assert end - start == pytest.approx(10, abs=0.02)

    assert place_window(0.1, 0.4, 10, 2) == pytest.approx((0.0, 2.0), abs=0.01)
    start, end = place_window(9.7, 9.9, 10, 2)
    assert start == pytest.approx(8.0, abs=0.01)
    assert end == pytest.approx(10.0, abs=0.01)

    assert choose_duration(3.2, [2, 4, 6], 30, 6) == 4
    with pytest.raises(RuntimeError, match="shortest"):
        choose_duration(0.5, [2, 4], 1.5, 4)


def test_poll_retries_transient_errors_and_stops_on_terminal_states():
    state, now, sleep = _clock()

    def once(outcome):
        calls = {"n": 0}

        def transport(method, url, payload):
            calls["n"] += 1
            assert method == "GET"
            if isinstance(outcome, Exception):
                raise outcome
            return outcome

        return transport, calls

    transport, calls = once(HttpError(400, "bad duration"))
    with pytest.raises(HttpError, match="400"):
        poll_video("job", transport, sleep=sleep, clock=now, deadline=100, interval=20)
    assert calls["n"] == 1
    assert state["t"] == 0

    for status in ("failed", "cancelled", "expired"):
        transport, calls = once({"status": status, "error": status})
        with pytest.raises(RuntimeError, match=status):
            poll_video("job", transport, sleep=sleep, clock=now, deadline=100, interval=20)
        assert calls["n"] == 1

    seen = []

    def transport(method, url, payload):
        seen.append(state["t"])
        if len(seen) == 1:
            raise HttpError(429, "slow")
        if len(seen) == 2:
            raise HttpError(503, "down")
        if len(seen) == 3:
            raise ConnectionError("reset")
        if len(seen) == 4:
            raise HttpError(408, "late")
        return {"status": "in_progress", "progress": 40}

    state["t"] = 0
    with pytest.raises(TimeoutError):
        poll_video("job", transport, sleep=sleep, clock=now, deadline=120, interval=20)
    assert seen[:4] == [0, 20, 40, 60]
    assert state["t"] >= 120

    seen = {"urls": []}

    def failing(method, url, payload):
        seen["urls"].append(url)
        if method == "POST":
            return {"id": "job-x", "status": "pending"}
        if "content" in url:
            raise AssertionError("downloaded a failed job")
        return {"id": "job-x", "status": "failed", "error": "provider"}

    with pytest.raises(RuntimeError, match="failed"):
        generate_clip({"model": "alibaba/wan-3.0"}, failing, sleep=sleep, clock=now, deadline=1000, interval=20)
    assert not any("content" in url for url in seen["urls"])


def test_a_running_job_is_polled_and_not_resubmitted():
    calls = []
    posts = [{"id": "job-7", "status": "pending"}]
    polls = [
        HttpError(429, "slow"),
        HttpError(502, "down"),
        ConnectionError("reset"),
        HttpError(408, "late"),
        {"id": "job-7", "status": "in_progress", "progress": 30},
        {
            "id": "job-7",
            "status": "completed",
            "unsigned_urls": ["/api/v1/videos/job-7/content?index=0"],
        },
    ]
    video = b"\x00\x00\x00\x18ftypmp42"
    downloads = [HttpError(503, "later"), video]

    def transport(method, url, payload):
        calls.append((method, url, payload))
        if method == "POST":
            assert url == API
            item = posts.pop(0)
        elif "/content" in url:
            assert url == "https://openrouter.ai/api/v1/videos/job-7/content?index=0"
            item = downloads.pop(0)
        else:
            assert url == f"{API}/job-7"
            item = polls.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    state, now, sleep = _clock()
    body = video_request("Replace the bottle", "https://media.example/window.mp4", 2, 3)
    data = generate_clip(body, transport, sleep=sleep, clock=now, deadline=10_000, interval=20)
    assert data == video
    assert [call[0] for call in calls].count("POST") == 1
    assert calls[0][0] == "POST"
    assert calls[0][2]["model"] == "alibaba/wan-3.0"
    assert calls[0][2]["generate_audio"] is False
    assert state["t"] > 0
    assert posts == [] and polls == [] and downloads == []

    state["t"] = 0
    saved = {}

    def down(method, url, payload):
        if state["t"] == 0:
            raise HttpError(503, "later")
        saved["url"] = url
        return video

    raw = download_output(
        {"unsigned_urls": ["https://cdn.example/out.mp4"]},
        down,
        sleep=sleep,
        clock=now,
        deadline=100,
        interval=20,
    )
    assert raw == video
    assert saved["url"] == "https://cdn.example/out.mp4"


def test_poll_budget_is_the_longer_of_twelve_minutes_or_one_and_a_half():
    assert poll_budget_s(2) == 12 * 60
    assert poll_budget_s(10) == 12 * 60
    assert poll_budget_s(30) == 36 * 60


def test_submit_is_not_retried():
    calls = {"n": 0}

    def broken(method, url, payload):
        calls["n"] += 1
        raise ConnectionError("reset")

    with pytest.raises(ConnectionError):
        submit_video({"model": "alibaba/wan-3.0"}, broken)
    assert calls["n"] == 1

    calls["n"] = 0

    def limited(method, url, payload):
        calls["n"] += 1
        raise HttpError(503, "down")

    with pytest.raises(HttpError):
        submit_video({"model": "alibaba/wan-3.0"}, limited)
    assert calls["n"] == 1


def test_a_second_swap_is_rejected_while_one_is_queued(tmp_path):
    video = tmp_path / "clip.mp4"
    video.write_bytes(b"x")
    save_findings(
        str(video),
        [
            {
                "id": "obj1",
                "label": "cup",
                "category": "other",
                "start_s": 0.0,
                "end_s": 1.0,
                "visibility": 1.0,
                "boxes": [],
            }
        ],
        0,
        root=tmp_path,
    )
    clip = list_clips(tmp_path)[0]
    hold = threading.Event()
    entered = threading.Event()

    def prepare():
        entered.set()
        assert hold.wait(5)

    deps = SwapDeps(prepare=prepare)
    kwargs = {
        "clip_id": clip["id"],
        "object_id": clip["objects"][0]["id"],
        "image": None,
        "root": tmp_path,
        "deps": deps,
    }
    first = start_swap(product="Starbucks cup", **kwargs)
    try:
        assert entered.wait(2)
        with pytest.raises(SwapBusy, match="already"):
            start_swap(product="Another cup", **kwargs)
        assert get_swap(first["id"])["status"] == "queued"
    finally:
        hold.set()

    def wait_done(swap_id: str) -> dict:
        deadline = time.monotonic() + 5
        job = get_swap(swap_id)
        while time.monotonic() < deadline:
            job = get_swap(swap_id)
            if job and job["status"] in {"failed", "completed"}:
                return job
            time.sleep(0.02)
        return job

    assert wait_done(first["id"])["status"] == "failed"
    second = start_swap(product="Another cup", **kwargs)
    assert second["status"] == "queued"
    assert wait_done(second["id"])["status"] == "failed"


def _counted(path: Path, color: str, size: str, rate: str, frames: int, audio: bool) -> None:
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"color=c={color}:s={size}:r={rate}",
    ]
    if audio:
        cmd += ["-f", "lavfi", "-i", "sine=frequency=440:r=48000"]
    cmd += ["-frames:v", str(frames), "-c:v", "libx264", "-pix_fmt", "yuv420p"]
    if audio:
        cmd += ["-c:a", "aac", "-shortest"]
    else:
        cmd += ["-an"]
    cmd.append(str(path))
    _ffmpeg(cmd)


def _frames(path: Path) -> tuple[int, float]:
    proc = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-count_frames",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_read_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(proc.stdout)["streams"][0]
    return int(stream["nb_read_frames"]), float(stream["duration"])


def test_finish_muxes_source_audio_and_matches_the_frame(tmp_path):
    plan = finish_plan(
        {"width": 64, "height": 64, "fps": 10, "fps_arg": "10/1", "has_audio": False},
        {"width": 64, "height": 64, "fps": 10, "fps_arg": "10/1", "has_audio": False},
    )
    assert plan == {"scale": None, "fps": None, "audio": False, "copy_video": True}
    quiet_window = tmp_path / "quiet-window.mp4"
    quiet_generated = tmp_path / "quiet-generated.mp4"
    quiet_out = tmp_path / "quiet-out.mp4"
    _color(quiet_window, "green", "64x64", 1, 10, audio=False)
    _color(quiet_generated, "blue", "64x64", 1, 10, audio=False)
    finish_clip(quiet_generated, quiet_window, quiet_out)
    quiet = _probe(quiet_out)
    assert quiet["has_audio"] is False
    assert (quiet["width"], quiet["height"]) == (64, 64)

    window = tmp_path / "window.mp4"
    generated = tmp_path / "generated.mp4"
    dest = tmp_path / "out.mp4"
    _color(window, "green", "320x180", 2, 25, audio=True)
    _color(generated, "blue", "128x72", 2, 12, audio=False)
    finish_clip(generated, window, dest)
    info = _probe(dest)
    assert info["has_audio"] is True
    assert (info["width"], info["height"]) == (320, 180)
    assert info["fps"] == pytest.approx(25, abs=0.1)
    assert info["duration"] == pytest.approx(2, abs=0.2)
    red, green, blue = _rgb(dest)
    assert blue > red + 20
    assert blue > green + 20

    tall = tmp_path / "tall.mp4"
    wide = tmp_path / "wide.mp4"
    fitted = tmp_path / "fitted.mp4"
    _color(tall, "green", "180x320", 1, 25, audio=False)
    _color(wide, "blue", "320x180", 1, 25, audio=False)
    finish_clip(wide, tall, fitted)
    info = _probe(fitted)
    assert (info["width"], info["height"]) == (180, 320)
    center = _pixel(fitted, 90, 160)
    edge = _pixel(fitted, 90, 4)
    assert center[2] > center[0] + 20
    assert max(edge) < 40


def test_finish_matches_the_window_frame_count(tmp_path):
    window = tmp_path / "film.mp4"
    generated = tmp_path / "wan.mp4"
    dest = tmp_path / "matched.mp4"
    _counted(window, "green", "64x64", "24000/1001", 143, audio=True)
    _counted(generated, "blue", "128x72", "30", 174, audio=False)
    finish_clip(generated, window, dest)
    window_frames, window_duration = _frames(window)
    out_frames, out_duration = _frames(dest)
    assert window_frames == 143
    assert out_frames == window_frames
    assert out_duration == pytest.approx(window_duration, abs=0.05)


def test_transport_sends_the_key_only_as_a_header(monkeypatch):
    sent = {}

    class Resp:
        headers = {"Content-Type": "application/json"}

        def read(self):
            return b'{"id":"job-1","status":"pending"}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def urlopen(req, timeout=0):
        sent["auth"] = req.get_header("Authorization")
        sent["url"] = req.full_url
        sent["data"] = req.data
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    body = urllib_transport("POST", API, {"model": "alibaba/wan-3.0"}, key=_KEY)
    assert body["id"] == "job-1"
    assert sent["auth"] == f"Bearer {_KEY}"
    assert _KEY not in sent["url"]
    assert _KEY not in sent["data"].decode()

    def boom(req, timeout=0):
        raise urllib.error.HTTPError(req.full_url, 401, "unauthorized", hdrs=None, fp=io.BytesIO(f"no {_KEY}".encode()))

    monkeypatch.setattr(urllib.request, "urlopen", boom)
    with pytest.raises(HttpError) as exc:
        urllib_transport("GET", f"{API}/job-1", None, key=_KEY)
    assert exc.value.code == 401
    assert _KEY not in str(exc.value)
    assert "[redacted]" in str(exc.value)


def test_errors_drop_the_key_and_the_presigned_query(monkeypatch):
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret-value")
    text = _public_error(
        RuntimeError(
            f"no {_KEY} https://bucket.example/a.mp4?X-Amz-Signature=aws-secret-value&X-Amz-Credential=abc"
        ),
        _KEY,
    )
    assert _KEY not in text
    assert "aws-secret-value" not in text
    assert "X-Amz-Signature" not in text
    assert "https://bucket.example/a.mp4" in text


def test_swap_returns_before_the_job_and_serves_the_result(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", _KEY)
    video = tmp_path / "clip.mp4"
    _color(video, "green", "320x180", 4, 25, audio=True)
    generated = tmp_path / "wan.mp4"
    _color(generated, "blue", "128x72", 2, 12, audio=False)
    save_findings(
        video,
        [
            {
                "label": "green wine bottle",
                "category": "bottle",
                "suggested_product": "branded wine bottle",
                "start_s": 1.0,
                "end_s": 1.5,
                "visibility": 0.9,
                "boxes": [{"t": 1.2, "x": 200, "y": 70, "w": 40, "h": 40}],
            }
        ],
        0.01,
        root=tmp_path,
    )
    clip = list_clips(tmp_path)[0]
    obj = clip["objects"][0]
    hold = threading.Event()
    entered = threading.Event()
    seen = {"calls": [], "uploads": []}

    def prepare():
        entered.set()
        assert hold.wait(5)

    def upload(path, key, content_type):
        seen["uploads"].append((key, content_type, path.stat().st_size))
        return "https://media.example/" + key

    def transport(method, url, payload):
        seen["calls"].append((method, url, payload))
        if method == "GET" and url.endswith("/models"):
            return {"data": [{"id": "alibaba/wan-3.0", "supported_durations": [2, 4, 6, 8, 10]}]}
        if method == "POST":
            seen["body"] = payload
            return {"id": "job-1", "status": "pending"}
        if method == "GET" and url.endswith("/job-1"):
            return {
                "id": "job-1",
                "status": "completed",
                "progress": 100,
                "usage": {"cost": 0.72},
                "unsigned_urls": ["/api/v1/videos/job-1/content?index=0"],
            }
        if method == "GET" and "/content" in url:
            return generated.read_bytes()
        raise AssertionError((method, url))

    deps = SwapDeps(transport=transport, upload=upload, sleep=lambda _seconds: None, prepare=prepare)
    httpd = serve(tmp_path, port=0, deps=deps)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "Replace object" in page
        assert 'id="original"' in page
        assert 'id="result"' in page
        assert "job.cost" in page
        target = f"{base}/api/clips/{clip['id']}/objects/{obj['id']}/swap"

        def post(payload):
            req = urllib.request.Request(
                target,
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            return urllib.request.urlopen(req, timeout=2)

        with pytest.raises(urllib.error.HTTPError) as blank:
            post({"product": "   "})
        assert blank.value.code == 400
        blank.value.close()
        with pytest.raises(urllib.error.HTTPError) as bad_image:
            post({"product": "Acme Cola can", "image_base64": base64.b64encode(b"nope").decode()})
        assert bad_image.value.code == 400
        bad_image.value.close()

        started = time.monotonic()
        with post(
            {"product": "Acme Cola can", "image_base64": base64.b64encode(_PNG).decode()}
        ) as resp:
            assert resp.status == 202
            posted = json.load(resp)
        assert time.monotonic() - started < 1
        assert posted["status"] == "queued"
        assert _KEY not in json.dumps(posted)
        assert entered.wait(2)
        mid = json.load(urllib.request.urlopen(base + "/api/swaps/" + posted["id"]))
        assert mid["status"] == "queued"
        with pytest.raises(urllib.error.HTTPError) as busy:
            post({"product": "Second product"})
        assert busy.value.code == 409
        busy.value.close()
        hold.set()

        deadline = time.monotonic() + 30
        job = mid
        while time.monotonic() < deadline:
            job = json.load(urllib.request.urlopen(base + "/api/swaps/" + posted["id"]))
            if job["status"] in {"completed", "failed"}:
                break
            time.sleep(0.05)
        assert job["status"] == "completed", job
        assert job["cost"] == 0.72
        assert job["result_url"]
        assert job["window_url"]
        body = seen["body"]
        assert "green wine bottle" in body["prompt"]
        assert "center-right of the frame" in body["prompt"]
        assert "Acme Cola can" in body["prompt"]
        assert "lighting direction" in body["prompt"]
        assert "color temperature" in body["prompt"]
        assert body["model"] == "alibaba/wan-3.0"
        assert body["duration"] == 2
        assert body["resolution"] == "720p"
        assert body["aspect_ratio"] == "16:9"
        assert body["generate_audio"] is False
        assert body["input_references"][0]["video_url"]["url"].startswith("https://")
        assert body["input_references"][1]["type"] == "image_url"
        assert body["input_references"][1]["image_url"]["url"].startswith("https://")
        assert _KEY not in json.dumps(body)
        assert [call[0] for call in seen["calls"]].count("POST") == 1
        assert [item[1] for item in seen["uploads"]] == ["video/mp4", "image/png"]
        listed = json.load(urllib.request.urlopen(f"{base}/api/clips/{clip['id']}/swaps"))
        assert [item["id"] for item in listed] == [posted["id"]]
        window_bytes = urllib.request.urlopen(base + job["window_url"]).read()
        result_bytes = urllib.request.urlopen(base + job["result_url"]).read()
        assert len(window_bytes) > 100
        result_path = tmp_path / "served.mp4"
        result_path.write_bytes(result_bytes)
        info = _probe(result_path)
        assert info["has_audio"] is True
        assert (info["width"], info["height"]) == (320, 180)
        red, green, blue = _rgb(result_path)
        assert blue > red + 20 and blue > green + 20
    finally:
        hold.set()
        httpd.shutdown()
