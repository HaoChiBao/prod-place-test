"""Replace one detected object with a branded product.

The object's span is cut with ffmpeg, padded to at least two seconds, and clamped
to the Wan 3.0 maximum from the video models list. The clip and an optional product
image are uploaded to S3. Alibaba Wan 3.0 edits the clip through OpenRouter. The
result is fitted back to the window's frame rate and size, and the original audio
is muxed on when the source had any.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from find_objects import probe_video
from store import catalog_dir, get_clip

API = "https://openrouter.ai/api/v1/videos"
MODEL = "alibaba/wan-3.0"
RESOLUTION = "720p"
ASPECT_RATIO = "16:9"
_ASPECTS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9")
MIN_WINDOW_S = 2.0
POLL_EVERY_S = 20.0
PRESIGN_SECONDS = 3 * 60 * 60
MODELS_BUDGET_S = 120.0

_ACROSS = ("left", "center-left", "center", "center-right", "right")
_DOWN = ("top", "upper", "middle", "lower", "bottom")
_QUERY = re.compile(r"(https?://[^\s?]+)\?[^\s]*")
_RETRYABLE = {408, 429}
_TERMINAL = {"failed", "cancelled", "expired"}

Transport = Callable[[str, str, dict | None], Any]
Sleep = Callable[[float], None]
Clock = Callable[[], float]


class HttpError(Exception):
    def __init__(self, code: int, detail: str):
        self.code = code
        super().__init__(f"openrouter HTTP {code}: {detail}")


@dataclass
class SwapDeps:
    transport: Transport | None = None
    upload: Callable[[Path, str, str], str] | None = None
    sleep: Sleep | None = None
    clock: Clock | None = None
    prepare: Callable[[], None] | None = None
    interval: float = POLL_EVERY_S


_jobs: dict[str, dict] = {}
_lock = threading.Lock()


def position_phrase(box: dict, width: int, height: int) -> str:
    """Rough place of a pixel box, such as 'center-right of the frame'."""
    if width < 1 or height < 1:
        return "center of the frame"
    cx = (float(box["x"]) + float(box["w"]) / 2.0) / float(width)
    cy = (float(box["y"]) + float(box["h"]) / 2.0) / float(height)
    across = _ACROSS[_band(cx)]
    down = _DOWN[_band(cy)]
    if down == "middle" and across == "center":
        return "center of the frame"
    if down == "middle":
        return f"{across} of the frame"
    if across == "center":
        return f"{down} center of the frame"
    return f"{down} {across} of the frame"


def swap_prompt(label: str, position: str, product: str, *, reference_image: bool) -> str:
    label = _one_line(label) or "object"
    position = _one_line(position) or "center of the frame"
    product = _one_line(product)
    text = (
        f"Replace the {label} at {position} with {product}, in exactly the same position, "
        "size, orientation and hand placement. Make it photorealistic and physically integrated: "
        "match the scene's lighting direction, color temperature and intensity, cast matching shadows "
        "and contact shadows, show the same reflections and specular highlights as nearby glass and "
        "surfaces, match depth of field, motion blur, film grain and color grading, and keep occlusions "
        "by hands and other objects. Change nothing else: faces, people, motion, background, camera "
        "and shot cuts stay identical."
    )
    if reference_image:
        text += (
            f" The reference image shows only what {product} looks like. Keep it locked at {position}, "
            f"in the same place, size, and orientation as the {label}. Do not move it or restage the shot."
        )
    return text


def video_request(
    prompt: str,
    video_url: str,
    duration: int,
    seed: int,
    image_url: str | None = None,
    aspect_ratio: str = ASPECT_RATIO,
) -> dict:
    refs = [{"type": "video_url", "video_url": {"url": _https(video_url)}}]
    if image_url:
        refs.append({"type": "image_url", "image_url": {"url": _https(image_url)}})
    return {
        "model": MODEL,
        "prompt": prompt,
        "input_references": refs,
        "duration": int(duration),
        "resolution": RESOLUTION,
        "aspect_ratio": aspect_ratio,
        "generate_audio": False,
        "seed": int(seed),
    }


def choose_aspect(width: int, height: int, supported: list[str] | None = None) -> str:
    """The supported ratio closest to the frame, so the edit is not restaged."""
    options = [item for item in (supported or _ASPECTS) if _ratio(item) is not None]
    if not options or width < 1 or height < 1:
        return ASPECT_RATIO
    target = float(width) / float(height)
    return min(options, key=lambda item: abs(target - _ratio(item)))


def aspects_for(payload: Any) -> list[str]:
    models = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        return list(_ASPECTS)
    for model in models:
        if not isinstance(model, dict):
            continue
        if (model.get("id") or model.get("canonical_slug")) != MODEL:
            continue
        found = []
        for item in model.get("supported_aspect_ratios") or []:
            if isinstance(item, str) and _ratio(item.strip()) is not None:
                found.append(item.strip())
        return found or list(_ASPECTS)
    return list(_ASPECTS)


def durations_for(payload: Any) -> tuple[list[int], int]:
    """Supported whole-second durations for Wan 3.0, and their maximum."""
    models = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(models, list):
        raise RuntimeError("openrouter models response")
    for model in models:
        if not isinstance(model, dict):
            continue
        ident = model.get("id") or model.get("canonical_slug")
        if ident != MODEL:
            continue
        supported = _duration_list(model.get("supported_durations"))
        if not supported:
            raise RuntimeError("alibaba/wan-3.0 has no supported durations")
        return supported, supported[-1]
    raise RuntimeError("alibaba/wan-3.0 is not in the video models list")


def choose_duration(span_s: float, supported: list[int], video_duration: float, model_max: int) -> int:
    """Shortest supported length that covers the span, at least two seconds, at most the model max."""
    options = [d for d in supported if d <= model_max and d <= float(video_duration) + 1e-3]
    if not options:
        shortest = min(supported) if supported else None
        if shortest is not None and float(video_duration) + 1e-3 < shortest:
            raise RuntimeError(f"the clip is shorter than the model's shortest video ({shortest}s)")
        raise RuntimeError("no supported duration fits this video")
    limit = min(float(model_max), float(video_duration))
    target = min(max(float(span_s), MIN_WINDOW_S), limit)
    covering = [d for d in options if d + 1e-6 >= target]
    if covering:
        return min(covering)
    return max(options)


def place_window(start_s: float, end_s: float, video_duration: float, duration_s: float) -> tuple[float, float]:
    """A window of ``duration_s`` seconds around the object, kept inside the video."""
    video = float(video_duration)
    duration = float(duration_s)
    if duration <= 0 or video <= 0:
        raise RuntimeError("empty window")
    if duration > video + 1e-3:
        raise RuntimeError("window is longer than the video")
    obj_start = min(max(0.0, float(start_s)), video)
    obj_end = min(max(obj_start, float(end_s)), video)
    start = (obj_start + obj_end) / 2.0 - duration / 2.0
    if obj_end - obj_start <= duration + 1e-6:
        if start > obj_start:
            start = obj_start
        if start + duration < obj_end:
            start = obj_end - duration
    if start < 0:
        start = 0.0
    if start + duration > video:
        start = video - duration
    start = round(max(0.0, start), 3)
    return start, round(start + duration, 3)


def poll_budget_s(duration_s: float) -> float:
    """About eight minutes for each ten seconds of generated video."""
    return max(1.0, float(duration_s) / 10.0) * 8.0 * 60.0


def submit_video(
    body: dict,
    transport: Transport,
    *,
    sleep: Sleep,
    clock: Clock,
    deadline: float,
    interval: float = POLL_EVERY_S,
) -> str:
    def op() -> str:
        data = transport("POST", API, body)
        if not isinstance(data, dict) or not str(data.get("id") or "").strip():
            raise RuntimeError("openrouter response missing id")
        return str(data["id"]).strip()

    return _retry(op, sleep=sleep, clock=clock, deadline=deadline, interval=interval)


def poll_video(
    job_id: str,
    transport: Transport,
    *,
    sleep: Sleep,
    clock: Clock,
    deadline: float,
    on_status: Callable[[dict], None] | None = None,
    interval: float = POLL_EVERY_S,
) -> dict:
    url = f"{API}/{job_id}"

    def op() -> dict:
        data = transport("GET", url, None)
        if not isinstance(data, dict):
            raise RuntimeError("openrouter response was not JSON")
        return data

    while True:
        data = _retry(op, sleep=sleep, clock=clock, deadline=deadline, interval=interval)
        status = str(data.get("status") or "")
        if on_status is not None and status not in _TERMINAL:
            on_status(data)
        if status == "completed":
            return data
        if status in _TERMINAL:
            detail = data.get("error") or status
            raise RuntimeError(f"video job {status}: {detail}")
        if clock() >= deadline:
            raise TimeoutError("openrouter timed out")
        _pause(sleep, clock, deadline, interval)


def download_output(
    job: dict,
    transport: Transport,
    *,
    sleep: Sleep,
    clock: Clock,
    deadline: float,
    interval: float = POLL_EVERY_S,
) -> bytes:
    urls = job.get("unsigned_urls") if isinstance(job, dict) else None
    if not isinstance(urls, list) or not urls:
        raise RuntimeError("openrouter response missing unsigned_urls")
    url = _absolute(str(urls[0]))

    def op() -> bytes:
        data = transport("GET", url, None)
        if not isinstance(data, (bytes, bytearray)) or not data:
            raise RuntimeError("openrouter download was empty")
        return bytes(data)

    return _retry(op, sleep=sleep, clock=clock, deadline=deadline, interval=interval)


def generate_clip(
    body: dict,
    transport: Transport,
    *,
    sleep: Sleep,
    clock: Clock,
    deadline: float,
    on_status: Callable[[dict], None] | None = None,
    interval: float = POLL_EVERY_S,
) -> bytes:
    """Submit once, then only poll and download that job."""
    job_id = submit_video(
        body, transport, sleep=sleep, clock=clock, deadline=deadline, interval=interval
    )
    completed = poll_video(
        job_id,
        transport,
        sleep=sleep,
        clock=clock,
        deadline=deadline,
        on_status=on_status,
        interval=interval,
    )
    return download_output(
        completed, transport, sleep=sleep, clock=clock, deadline=deadline, interval=interval
    )


def finish_plan(source: dict, generated: dict) -> dict:
    scale = None
    if source["width"] != generated["width"] or source["height"] != generated["height"]:
        scale = (int(source["width"]), int(source["height"]))
    fps = None
    src_fps = float(source.get("fps") or 0)
    gen_fps = float(generated.get("fps") or 0)
    if src_fps > 0 and gen_fps > 0 and abs(src_fps - gen_fps) > 0.05:
        fps = str(source.get("fps_arg") or src_fps)
    return {
        "scale": scale,
        "fps": fps,
        "audio": bool(source.get("has_audio")),
        "copy_video": scale is None and fps is None,
    }


def finish_clip(generated: Path, window: Path, dest: Path) -> None:
    """Match the generated picture to the window, and mux the window's audio."""
    source = _streams(window)
    plan = finish_plan(source, _streams(generated))
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", "-i", str(generated)]
    if plan["audio"]:
        cmd += ["-i", str(window)]
    filters = []
    if plan["scale"] is not None:
        width, height = plan["scale"]
        filters.append(
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2"
        )
    if plan["fps"] is not None:
        filters.append(f"fps={plan['fps']}")
    if filters:
        cmd += ["-vf", ",".join(filters)]
    cmd += ["-map", "0:v:0"]
    if plan["audio"]:
        cmd += ["-map", "1:a:0", "-c:a", "copy"]
    if plan["copy_video"]:
        cmd += ["-c:v", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-pix_fmt", "yuv420p"]
    cmd += ["-movflags", "+faststart", "-shortest", str(dest)]
    _run(cmd)


def cut_window(source: Path, dest: Path, start_s: float, duration_s: float) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    info = _streams(source)
    cmd = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(source),
        "-ss",
        f"{start_s:.3f}",
        "-t",
        f"{duration_s:.3f}",
        "-map",
        "0:v:0",
    ]
    if info["has_audio"]:
        cmd += ["-map", "0:a:0", "-c:a", "aac", "-b:a", "128k"]
    cmd += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(dest),
    ]
    _run(cmd)


def swap_seed(*parts: str) -> int:
    value = zlib.crc32("\0".join(parts).encode()) & 0x7FFFFFFF
    return value or 1


def s3_upload(path: Path, object_key: str, content_type: str) -> str:
    bucket = (os.environ.get("AWS_S3_BUCKET_NAME") or "").strip()
    access = (os.environ.get("AWS_ACCESS_KEY_ID") or "").strip()
    secret = (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    region = (os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "us-east-1").strip()
    if not bucket or not access or not secret:
        raise RuntimeError("AWS_S3_BUCKET_NAME, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY must be set")
    import boto3

    client = boto3.client(
        "s3",
        region_name=region,
        aws_access_key_id=access,
        aws_secret_access_key=secret,
    )
    client.upload_file(str(path), bucket, object_key, ExtraArgs={"ContentType": content_type})
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": object_key},
        ExpiresIn=PRESIGN_SECONDS,
    )
    if not isinstance(url, str) or not url.startswith("https://"):
        raise RuntimeError("presigned url is not https")
    return url


def urllib_transport(method: str, url: str, payload: dict | None, *, key: str) -> Any:
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Accept": "*/*",
        },
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            raw = resp.read()
            ctype = resp.headers.get("Content-Type") or ""
    except urllib.error.HTTPError as exc:
        detail = _redact(exc.read().decode("utf-8", "replace"), key)
        raise HttpError(exc.code, detail) from None
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
        raise ConnectionError(_redact(str(reason), key)) from None
    if "json" in ctype or (raw[:1] in (b"{", b"[") and "video" not in ctype and "octet-stream" not in ctype):
        try:
            return json.loads(raw.decode())
        except json.JSONDecodeError:
            raise RuntimeError("openrouter response was not JSON") from None
    return raw


def start_swap(
    *,
    clip_id: str,
    object_id: str,
    product: str,
    image: bytes | None,
    root: Path | None,
    deps: SwapDeps | None = None,
) -> dict:
    job_id = uuid.uuid4().hex[:12]
    job = {
        "id": job_id,
        "clip_id": clip_id,
        "object_id": object_id,
        "product": _one_line(product),
        "label": "",
        "status": "queued",
        "detail": "",
        "progress": None,
        "error": None,
        "window_ready": False,
        "created": time.time(),
        "_image": image,
    }
    with _lock:
        _jobs[job_id] = job
        public = _public(job)
    threading.Thread(
        target=_worker,
        args=(job_id, root, deps or SwapDeps()),
        daemon=True,
    ).start()
    return public


def get_swap(swap_id: str) -> dict | None:
    with _lock:
        job = _jobs.get(swap_id)
        return None if job is None else _public(job)


def list_swaps(clip_id: str) -> list[dict]:
    with _lock:
        jobs = [job for job in _jobs.values() if job["clip_id"] == clip_id]
        jobs.sort(key=lambda job: job["created"])
        return [_public(job) for job in jobs]


def swap_path(root: Path | None, swap_id: str, kind: str) -> Path | None:
    name = {"window": "window.mp4", "result": "result.mp4"}.get(kind)
    if name is None:
        return None
    path = catalog_dir(root) / "swaps" / swap_id / name
    if not path.is_file():
        return None
    return path


def reset_jobs() -> None:
    with _lock:
        _jobs.clear()


def execute_swap(job_id: str, *, root: Path | None, deps: SwapDeps, key: str) -> None:
    transport = deps.transport
    if transport is None:
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")
        transport = lambda method, url, payload: urllib_transport(method, url, payload, key=key)
    upload = deps.upload or s3_upload
    sleep = deps.sleep or time.sleep
    clock = deps.clock or time.monotonic
    interval = deps.interval

    with _lock:
        job = _jobs[job_id]
        image = job.pop("_image", None)
        clip_id = job["clip_id"]
        object_id = job["object_id"]
        product = job["product"]

    clip = get_clip(clip_id, root)
    if clip is None:
        raise RuntimeError("clip not found")
    obj = next((item for item in clip["objects"] if item["id"] == object_id), None)
    if obj is None:
        raise RuntimeError("object not found")
    _update(job_id, status="running", label=str(obj.get("label") or "object"), detail="models")
    video = Path(clip["video_path"])
    if not video.is_file():
        raise RuntimeError("video missing")

    model_deadline = clock() + MODELS_BUDGET_S
    supported, model_max, aspects = _model_limits(
        transport, sleep=sleep, clock=clock, deadline=model_deadline, interval=interval
    )
    duration_s, width, height = probe_video(video)
    span = float(obj["end_s"]) - float(obj["start_s"])
    duration = choose_duration(span, supported, duration_s, model_max)
    start, _end = place_window(float(obj["start_s"]), float(obj["end_s"]), duration_s, duration)

    _update(job_id, detail="cutting")
    folder = catalog_dir(root) / "swaps" / job_id
    window = folder / "window.mp4"
    cut_window(video, window, start, duration)
    _update(job_id, window_ready=True, detail="uploading")

    video_url = upload(window, f"swaps/{job_id}/window.mp4", "video/mp4")
    _https(video_url)
    image_url = None
    if image:
        suffix, content_type = sniff_image(image)
        image_path = folder / f"product{suffix}"
        image_path.write_bytes(image)
        image_url = upload(image_path, f"swaps/{job_id}/product{suffix}", content_type)
        _https(image_url)

    box = _box_for(obj)
    position = position_phrase(box, width, height) if box else "center of the frame"
    prompt = swap_prompt(str(obj.get("label") or "object"), position, product, reference_image=image_url is not None)
    aspect = choose_aspect(width, height, aspects)
    body = video_request(
        prompt,
        video_url,
        duration,
        swap_seed(clip_id, object_id, product),
        image_url,
        aspect_ratio=aspect,
    )
    deadline = clock() + poll_budget_s(duration)
    _update(job_id, detail="submitting")

    def on_status(data: dict) -> None:
        fields: dict[str, Any] = {"detail": str(data.get("status_detail") or data.get("status") or "generating")}
        progress = _progress(data)
        if progress is not None:
            fields["progress"] = progress
        _update(job_id, **fields)

    raw = generate_clip(
        body,
        transport,
        sleep=sleep,
        clock=clock,
        deadline=deadline,
        on_status=on_status,
        interval=interval,
    )
    _update(job_id, detail="finishing")
    generated = folder / "generated.mp4"
    generated.write_bytes(raw)
    result = folder / "result.mp4"
    finish_clip(generated, window, result)
    if not result.is_file() or result.stat().st_size == 0:
        raise RuntimeError("swap produced no video")
    _update(job_id, status="completed", detail="", progress=100, error=None)


def sniff_image(data: bytes) -> tuple[str, str]:
    if data.startswith(b"\xff\xd8\xff"):
        return ".jpg", "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return ".png", "image/png"
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return ".webp", "image/webp"
    raise ValueError("image must be jpeg, png, or webp")


def decode_image(payload: dict) -> bytes | None:
    raw_b64 = payload.get("image_base64")
    if raw_b64 in (None, ""):
        return None
    if not isinstance(raw_b64, str):
        raise ValueError("image must be jpeg, png, or webp")
    import base64

    try:
        data = base64.b64decode(raw_b64, validate=True)
    except ValueError:
        raise ValueError("image must be jpeg, png, or webp") from None
    if len(data) > 8_000_000:
        raise ValueError("image is too large")
    sniff_image(data)
    return data


def _worker(job_id: str, root: Path | None, deps: SwapDeps) -> None:
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    try:
        if deps.prepare is not None:
            deps.prepare()
        execute_swap(job_id, root=root, deps=deps, key=key)
    except Exception as exc:
        _update(job_id, status="failed", error=_public_error(exc, key), detail="")


def _model_limits(transport: Transport, *, sleep: Sleep, clock: Clock, deadline: float, interval: float):
    def op():
        data = transport("GET", f"{API}/models", None)
        supported, model_max = durations_for(data)
        return supported, model_max, aspects_for(data)

    return _retry(op, sleep=sleep, clock=clock, deadline=deadline, interval=interval)


def _retry(op: Callable[[], Any], *, sleep: Sleep, clock: Clock, deadline: float, interval: float) -> Any:
    while True:
        if clock() >= deadline:
            raise TimeoutError("openrouter timed out")
        try:
            return op()
        except HttpError as exc:
            if exc.code not in _RETRYABLE and exc.code < 500:
                raise
        except (ConnectionError, TimeoutError, OSError):
            pass
        if clock() >= deadline:
            raise TimeoutError("openrouter timed out")
        _pause(sleep, clock, deadline, interval)


def _pause(sleep: Sleep, clock: Clock, deadline: float, interval: float) -> None:
    remaining = deadline - clock()
    if remaining > 0:
        sleep(min(interval, remaining))


def _public(job: dict) -> dict:
    ready = job["status"] == "completed"
    return {
        "id": job["id"],
        "clip_id": job["clip_id"],
        "object_id": job["object_id"],
        "product": job["product"],
        "label": job["label"],
        "status": job["status"],
        "detail": job["detail"],
        "progress": job["progress"],
        "error": job["error"],
        "window_url": f"/swaps/{job['id']}/window" if job["window_ready"] else None,
        "result_url": f"/swaps/{job['id']}/result" if ready else None,
    }


def _update(job_id: str, **fields: Any) -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def _box_for(obj: dict) -> dict | None:
    boxes = [box for box in (obj.get("boxes") or []) if isinstance(box, dict) and "x" in box]
    if not boxes:
        return None
    mid = (float(obj["start_s"]) + float(obj["end_s"])) / 2.0
    return min(boxes, key=lambda box: abs(float(box.get("t") or mid) - mid))


def _streams(path: Path) -> dict:
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type,width,height,avg_frame_rate,r_frame_rate",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    video = None
    has_audio = False
    for stream in data.get("streams") or []:
        if stream.get("codec_type") == "audio":
            has_audio = True
        elif stream.get("codec_type") == "video" and video is None:
            video = stream
    if not video or not video.get("width") or not video.get("height"):
        raise RuntimeError(f"no video stream in {path}")
    fps, fps_arg = _fps(video)
    return {
        "width": int(video["width"]),
        "height": int(video["height"]),
        "fps": fps,
        "fps_arg": fps_arg,
        "has_audio": has_audio,
    }


def _fps(stream: dict) -> tuple[float, str]:
    for key in ("avg_frame_rate", "r_frame_rate"):
        raw = stream.get(key)
        if not isinstance(raw, str) or raw in {"0/0", "N/A", ""}:
            continue
        num, sep, den = raw.partition("/")
        try:
            numerator = float(num)
            denominator = float(den) if den else 1.0
        except ValueError:
            continue
        if denominator and numerator > 0:
            return numerator / denominator, raw if sep else f"{numerator:g}"
    return 0.0, ""


def _run(args: list[str]) -> subprocess.CompletedProcess[str]:
    try:
        proc = subprocess.run(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            stdin=subprocess.DEVNULL,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"{args[0]} is not installed") from exc
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip().splitlines()
        tail = detail[-1] if detail else f"exit {proc.returncode}"
        raise RuntimeError(f"{args[0]} failed: {tail}")
    return proc


def _duration_list(raw: Any) -> list[int]:
    found: list[int] = []
    if not isinstance(raw, list):
        return found
    for item in raw:
        if isinstance(item, bool):
            continue
        if isinstance(item, int):
            number = item
        elif isinstance(item, float) and item.is_integer():
            number = int(item)
        elif isinstance(item, str) and item.strip().lstrip("-").isdigit():
            number = int(item.strip())
        else:
            continue
        if number > 0:
            found.append(number)
    return sorted(set(found))


def _ratio(text: str) -> float | None:
    left, sep, right = str(text).partition(":")
    if not sep:
        return None
    try:
        width = float(left)
        height = float(right)
    except ValueError:
        return None
    if width <= 0 or height <= 0:
        return None
    return width / height


def _band(value: float) -> int:
    clamped = min(0.999, max(0.0, value))
    return min(4, int(clamped * 5))


def _https(url: str) -> str:
    if not isinstance(url, str) or not url.startswith("https://"):
        raise ValueError("input url must be https")
    return url


def _absolute(url: str) -> str:
    if url.startswith("https://"):
        return url
    if url.startswith("/"):
        return "https://openrouter.ai" + url
    raise RuntimeError("unsigned url is not https")


def _one_line(text: str) -> str:
    return " ".join(str(text).split())


def _progress(data: dict) -> int | None:
    value = data.get("progress")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(max(0, min(100, round(value))))


def _public_error(exc: BaseException, key: str) -> str:
    text = _redact(str(exc), key)
    return text or exc.__class__.__name__


def _redact(text: str, key: str) -> str:
    flat = " ".join(str(text).split())
    if key:
        flat = flat.replace(key, "[redacted]")
    secret = (os.environ.get("AWS_SECRET_ACCESS_KEY") or "").strip()
    if secret:
        flat = flat.replace(secret, "[redacted]")
    flat = _QUERY.sub(r"\1", flat)
    return flat[:300]
