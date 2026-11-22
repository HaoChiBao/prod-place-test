"""Find existing on-screen objects that could be swapped for a branded product.

Shots come from ffmpeg scene cuts. Each shot contributes about one full-resolution
keyframe per second. Those frames go to Gemini in small batches. Returned boxes are
0-1000 [ymin, xmin, ymax, xmax] and are converted with the frame size.
"""

from __future__ import annotations

import base64
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "google/gemini-3.8-flash"
FRAMES_PER_CALL = 8
MIN_KEYFRAMES = 3
KEYFRAME_EVERY_S = 1.0
CUT_MARGIN_S = 0.15
VISIBILITY_MIN = 0.3

_PTS_TIME = re.compile(r"(?<![\w])pts_time:([0-9]+(?:\.[0-9]+)?)")
_FENCE = re.compile(r"^\s*```(?:json)?\s*(.*?)\s*```\s*$", re.IGNORECASE | re.DOTALL)
_STOP = frozenset({"a", "an", "the", "of", "on", "in", "and", "with"})
_SURFACE = frozenset(
    {
        "table",
        "tabletop",
        "surface",
        "counter",
        "countertop",
        "desk",
        "shelf",
        "floor",
        "wall",
        "background",
        "empty",
        "seat",
        "sofa",
        "couch",
        "bench",
        "place",
    }
)
_PEOPLE = frozenset({"person", "people", "face", "clothing", "clothes", "apparel"})

Request = Callable[[dict], Any]


def probe_video(path: Path) -> tuple[float, int, int]:
    """Duration in seconds, then width and height of the first video stream."""
    proc = _run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "format=duration:stream=width,height,duration",
            "-of",
            "json",
            str(path),
        ]
    )
    data = json.loads(proc.stdout)
    streams = data.get("streams") or []
    if not streams:
        raise RuntimeError(f"no video stream in {path}")
    stream = streams[0]
    duration = _video_duration(data)
    if duration is None:
        raise RuntimeError(f"no duration for {path}")
    return duration, int(stream["width"]), int(stream["height"])


def scene_cuts(path: Path) -> list[float]:
    """Cut times from ffmpeg select='gt(scene,0.3)',showinfo, read as pts_time."""
    proc = _run(
        [
            "ffmpeg",
            "-hide_banner",
            "-i",
            str(path),
            "-vf",
            "select='gt(scene,0.3)',showinfo",
            "-f",
            "null",
            "-",
        ]
    )
    text = proc.stderr + proc.stdout
    return [float(match) for match in _PTS_TIME.findall(text)]


def shot_ranges(cuts: list[float], duration: float) -> list[tuple[float, float]]:
    """Shots are the ranges between cut times, from 0 to the end of the video."""
    if duration <= 0:
        return []
    points = [0.0]
    for cut in sorted(float(cut) for cut in cuts):
        if cut <= points[-1] + 1e-3 or cut >= duration - 1e-3:
            continue
        points.append(cut)
    points.append(float(duration))
    return [(points[i], points[i + 1]) for i in range(len(points) - 1)]


def keyframe_times(start_s: float, end_s: float) -> list[float]:
    """About one frame per second, at least three, each 0.15s clear of a cut."""
    start, end = float(start_s), float(end_s)
    span = end - start
    if span <= 0:
        return []
    # Shorter than 0.3s cannot place a frame 0.15s clear of both cuts.
    if span < 2 * CUT_MARGIN_S:
        return []
    if span <= 2 * CUT_MARGIN_S:
        return [round((start + end) / 2.0, 3)]
    count = max(MIN_KEYFRAMES, int(round(span / KEYFRAME_EVERY_S)))
    usable = span - 2 * CUT_MARGIN_S
    times: list[float] = []
    for k in range(count):
        t = round(start + CUT_MARGIN_S + usable * k / (count - 1), 3)
        if not times or t > times[-1]:
            times.append(t)
    return times


def box_2d_to_xywh(box_2d: Any, width: int, height: int) -> dict[str, int]:
    """[ymin, xmin, ymax, xmax] on a 0-1000 scale to pixel x, y, w, h.

    Values are clamped to 0-1000. A 0-1 fraction is not rescaled.
    """
    if width < 1 or height < 1:
        raise ValueError("frame size")
    if not isinstance(box_2d, (list, tuple)) or len(box_2d) != 4:
        raise ValueError("box_2d")
    nums = [_finite(v) for v in box_2d]
    if any(n is None for n in nums):
        raise ValueError("box_2d")
    ymin, xmin, ymax, xmax = (min(1000.0, max(0.0, float(n))) for n in nums)
    if xmax <= xmin or ymax <= ymin:
        raise ValueError("empty box")
    x = int(round(xmin / 1000.0 * width))
    y = int(round(ymin / 1000.0 * height))
    x2 = int(round(xmax / 1000.0 * width))
    y2 = int(round(ymax / 1000.0 * height))
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    return {
        "x": x,
        "y": y,
        "w": max(1, min(width - x, x2 - x)),
        "h": max(1, min(height - y, y2 - y)),
    }


def merge_objects(
    detections: list[dict],
    shots: list[tuple[float, float]],
    keyframes: list[list[float]],
) -> list[dict]:
    """Merge same-category boxes that overlap or contain each other inside one shot.

    A food box and a plate that overlap in one frame keep the larger box.
    An object seen in only one keyframe with visibility below 0.3 is dropped.
    Start and end are the first and last keyframe it appears in. Appearance on
    the shot's first or last keyframe extends that edge to the shot boundary.
    """
    prepared = [_copy_det(det) for det in detections if _known_shot(det, shots)]
    prepared = _drop_smaller_food_or_plate(prepared)
    tracks: list[dict] = []
    ordered = sorted(prepared, key=lambda det: (det["shot_index"], det["t"], -det["visibility"]))
    for det in ordered:
        track = _matching_track(tracks, det)
        if track is None:
            tracks.append({"shot_index": det["shot_index"], "category": det["category"], "items": [det]})
        else:
            track["items"].append(det)

    found: list[dict] = []
    for track in tracks:
        items = _one_box_per_time(track["items"])
        if len(items) == 1 and items[0]["visibility"] < VISIBILITY_MIN:
            continue
        shot_index = track["shot_index"]
        start_s, end_s = _span(items, shots[shot_index], keyframes[shot_index])
        label, product = _label_and_product(items)
        visibility = sum(item["visibility"] for item in items) / len(items)
        found.append(
            {
                "label": label,
                "category": track["category"],
                "suggested_product": product,
                "shot_index": shot_index,
                "start_s": round(start_s, 3),
                "end_s": round(end_s, 3),
                "boxes": [
                    {
                        "t": round(item["t"], 3),
                        "x": item["box"]["x"],
                        "y": item["box"]["y"],
                        "w": item["box"]["w"],
                        "h": item["box"]["h"],
                    }
                    for item in items
                ],
                "visibility": round(visibility, 3),
            }
        )
    found.sort(key=lambda obj: (obj["shot_index"], obj["start_s"], obj["label"]))
    return found


def find_objects(
    video_path: str | Path,
    *,
    request: Request | None = None,
    usage: dict | None = None,
) -> list[dict]:
    """Return swappable objects with timestamps and pixel boxes.

    ``usage``, when passed, receives ``cost_usd`` and ``calls`` from OpenRouter.
    One batch that never returns JSON is skipped. Every batch failing raises.
    """
    path = Path(video_path)
    key = (os.environ.get("OPENROUTER_API_KEY") or "").strip()
    if request is None:
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is not set")

        def request(payload: dict, _key: str = key) -> Any:
            return _post_openrouter(payload, _key)

    duration, _, _ = probe_video(path)
    shots = shot_ranges(scene_cuts(path), duration)
    if not shots:
        if usage is not None:
            usage["cost_usd"] = 0.0
            usage["calls"] = 0
        return []
    per_shot = [keyframe_times(start, end) for start, end in shots]

    costs: list[float] = []
    with tempfile.TemporaryDirectory(prefix="swap-frames-") as tmp:
        frames = _extract_frames(path, per_shot, Path(tmp))
        detections: list[dict] = []
        batches = _chunks(frames, FRAMES_PER_CALL)
        failures = 0
        for batch in batches:
            try:
                items = _ask(batch, request, key, costs)
            except RuntimeError:
                failures += 1
                continue
            detections.extend(_detections_in_batch(items, batch))
    if usage is not None:
        usage["cost_usd"] = round(sum(costs), 6)
        usage["calls"] = len(costs)
    if batches and failures == len(batches):
        raise RuntimeError("product identification failed; no frame batch returned a valid answer")
    return merge_objects(detections, shots, per_shot)


def prompt_for(times: list[float]) -> str:
    listing = ", ".join(f"Frame {i} at {t:.3f}s" for i, t in enumerate(times))
    last = max(0, len(times) - 1)
    return (
        "List EXISTING visible objects that could be swapped for a branded product "
        "(bottles, cans, glasses, cups, devices, packaging, food, plates). "
        "Never list empty surfaces or places to put something. "
        "Box only the object itself. "
        f"There are {len(times)} frames in order ({listing}). "
        f"For each object return frame (0 to {last}), label, "
        "category (one word: bottle, can, glass, cup, device, packaging, food, plate, "
        "or another specific object word), "
        "suggested_product, visibility (0-1), and box_2d as "
        "[ymin, xmin, ymax, xmax] normalized 0-1000, not 0-1. "
        "Use category food for food and plate for the dish under it. "
        'JSON only: {"objects":[...]}.'
    )


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    save = False
    if args and args[0] == "--save":
        save = True
        args = args[1:]
    if len(args) != 1:
        print("usage: python find_objects.py [--save] VIDEO", file=sys.stderr)
        return 2
    spent: dict = {}
    found = find_objects(args[0], usage=spent)
    if save:
        from store import save_findings

        clip_id = save_findings(args[0], found, spent.get("cost_usd"))
        print(f"saved {clip_id}", file=sys.stderr)
    json.dump(found, sys.stdout, indent=2)
    sys.stdout.write("\n")
    cost = spent.get("cost_usd")
    if cost is not None:
        print(f"cost_usd {cost:.6f}", file=sys.stderr)
    return 0


def _ask(batch: list[dict], request: Request, key: str, costs: list[float] | None = None) -> list:
    payload = {
        "model": MODEL,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": prompt_for([frame["t"] for frame in batch])}]
                + [_image_part(frame["path"]) for frame in batch],
            }
        ],
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": "low"},
    }
    last = "model call failed"
    for _ in range(2):
        try:
            body = request(payload)
            if costs is not None:
                costs.append(_usage_cost(body))
            return _object_items(_message_text(body))
        except Exception as exc:
            last = str(exc)
    raise RuntimeError(_redact(last, key))


def _detections_in_batch(items: list, batch: list[dict]) -> list[dict]:
    found = []
    for item in items:
        if not isinstance(item, dict):
            continue
        index = _frame_index(item.get("frame"), len(batch))
        if index is None:
            continue
        frame = batch[index]
        label = _text(item.get("label"))
        category = _text(item.get("category")).lower() or "object"
        if not label or _is_surface(label, category):
            continue
        visibility = _finite(item.get("visibility"))
        if visibility is None:
            continue
        try:
            box = box_2d_to_xywh(item.get("box_2d"), frame["width"], frame["height"])
        except ValueError:
            continue
        found.append(
            {
                "shot_index": frame["shot_index"],
                "t": frame["t"],
                "label": label,
                "category": category,
                "suggested_product": _text(item.get("suggested_product")),
                "visibility": min(1.0, max(0.0, visibility)),
                "box": box,
            }
        )
    return found


def _extract_frames(video: Path, per_shot: list[list[float]], dest: Path) -> list[dict]:
    frames = []
    for shot_index, times in enumerate(per_shot):
        for t in times:
            path = dest / f"f{len(frames):04d}.jpg"
            _run(
                [
                    "ffmpeg",
                    "-y",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-i",
                    str(video),
                    "-map",
                    "0:v:0",
                    "-ss",
                    f"{t:.3f}",
                    "-frames:v",
                    "1",
                    "-q:v",
                    "2",
                    str(path),
                ]
            )
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"no frame extracted at {t:.3f}s")
            width, height = _jpeg_size(path.read_bytes())
            frames.append(
                {
                    "shot_index": shot_index,
                    "t": t,
                    "path": path,
                    "width": width,
                    "height": height,
                }
            )
    return frames


def _image_part(path: Path) -> dict:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{encoded}"}}


def _post_openrouter(payload: dict, key: str) -> Any:
    req = urllib.request.Request(
        OPENROUTER_URL,
        data=json.dumps(payload).encode(),
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            raw = resp.read().decode()
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"openrouter HTTP {exc.code}: {_redact(detail, key)}") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"openrouter request failed: {_redact(str(exc.reason), key)}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        raise RuntimeError("openrouter response was not JSON") from None


def _message_text(body: Any) -> str:
    if not isinstance(body, dict):
        raise RuntimeError("openrouter response")
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("openrouter response missing content") from exc
    if isinstance(content, str):
        text = content
    elif isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, str):
                parts.append(part)
            elif isinstance(part, dict) and isinstance(part.get("text"), str):
                kind = str(part.get("type") or "text").lower()
                if kind in {"reasoning", "thinking", "reasoning_text"}:
                    continue
                parts.append(part["text"])
        text = "".join(parts)
    else:
        raise RuntimeError("openrouter response missing content")
    if not text.strip():
        raise RuntimeError("empty content")
    return text


def _object_items(text: str) -> list:
    data = _parse_json(text)
    if isinstance(data, dict) and isinstance(data.get("objects"), list):
        return data["objects"]
    if isinstance(data, list) and all(isinstance(item, dict) for item in data):
        return data
    raise RuntimeError("missing objects")


def _parse_json(text: str) -> Any:
    raw = text.strip().lstrip("\ufeff")
    fenced = _FENCE.match(raw)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        extracted = _extract_json(raw)
        if extracted is None:
            raise RuntimeError("invalid JSON")
        return extracted


def _extract_json(text: str) -> Any | None:
    """The objects payload, not a preamble array such as a frame index or box_2d."""
    decoder = json.JSONDecoder()
    object_list = None
    index = 0
    while index < len(text):
        if text[index] not in "{[":
            index += 1
            continue
        try:
            value, consumed = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            index += 1
            continue
        if isinstance(value, dict) and isinstance(value.get("objects"), list):
            return value
        if object_list is None and isinstance(value, list) and all(isinstance(item, dict) for item in value):
            object_list = value
        index += max(consumed, 1)
    return object_list


def _video_duration(data: dict) -> float | None:
    """Video-stream length. The container duration includes a longer audio track."""
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    stream_len = _finite(stream.get("duration"))
    if stream_len is not None and stream_len > 0:
        return stream_len
    format_len = _finite((data.get("format") or {}).get("duration"))
    if format_len is not None and format_len > 0:
        return format_len
    return None


def _usage_cost(body: Any) -> float:
    if not isinstance(body, dict):
        return 0.0
    usage = body.get("usage")
    if not isinstance(usage, dict):
        return 0.0
    for name in ("cost", "total_cost"):
        number = _finite(usage.get(name))
        if number is not None and number >= 0:
            return number
    return 0.0


def _jpeg_size(data: bytes) -> tuple[int, int]:
    if len(data) < 4 or data[:2] != b"\xff\xd8":
        raise ValueError("not a jpeg")
    i = 2
    while i + 9 < len(data):
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker in (0xC0, 0xC1, 0xC2):
            height = int.from_bytes(data[i + 5 : i + 7], "big")
            width = int.from_bytes(data[i + 7 : i + 9], "big")
            if width < 1 or height < 1:
                break
            return width, height
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD9:
            i += 2
            continue
        if i + 4 > len(data):
            break
        seg = int.from_bytes(data[i + 2 : i + 4], "big")
        if seg < 2:
            break
        i += 2 + seg
    raise ValueError("jpeg size")


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


def _chunks(items: list, size: int) -> list[list]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    elif isinstance(value, str):
        try:
            number = float(value.strip())
        except ValueError:
            return None
    else:
        return None
    if not math.isfinite(number):
        return None
    return number


def _text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _frame_index(value: Any, count: int) -> int | None:
    if isinstance(value, bool):
        return None
    if value is None and count == 1:
        return 0
    number = _finite(value)
    if number is None or not number.is_integer():
        return None
    index = int(number)
    if index < 0 or index >= count:
        return None
    return index


def _is_surface(label: str, category: str) -> bool:
    if category in _SURFACE or category in _PEOPLE:
        return True
    tokens = [tok for tok in re.findall(r"[a-z0-9]+", label.lower()) if tok not in _STOP]
    return bool(tokens) and set(tokens) <= _SURFACE


def _redact(text: str, key: str) -> str:
    flat = " ".join(text.split())
    if key:
        flat = flat.replace(key, "[redacted]")
    return flat[:300]


def _copy_det(det: dict) -> dict:
    box = det["box"]
    return {
        "shot_index": int(det["shot_index"]),
        "t": float(det["t"]),
        "label": str(det["label"]).strip(),
        "category": str(det["category"]).strip().lower() or "object",
        "suggested_product": str(det.get("suggested_product") or "").strip(),
        "visibility": float(det["visibility"]),
        "box": {"x": int(box["x"]), "y": int(box["y"]), "w": int(box["w"]), "h": int(box["h"])},
    }


def _known_shot(det: dict, shots: list[tuple[float, float]]) -> bool:
    try:
        index = int(det["shot_index"])
    except (KeyError, TypeError, ValueError):
        return False
    return 0 <= index < len(shots)


def _area(box: dict) -> int:
    return int(box["w"]) * int(box["h"])


def _intersection(a: dict, b: dict) -> int:
    x1 = max(a["x"], b["x"])
    y1 = max(a["y"], b["y"])
    x2 = min(a["x"] + a["w"], b["x"] + b["w"])
    y2 = min(a["y"] + a["h"], b["y"] + b["h"])
    if x2 <= x1 or y2 <= y1:
        return 0
    return (x2 - x1) * (y2 - y1)


def _overlaps_or_contains(a: dict, b: dict) -> bool:
    """Intersecting boxes, including a box that sits inside the other."""
    return _intersection(a, b) > 0


def _drop_smaller_food_or_plate(dets: list[dict]) -> list[dict]:
    groups: dict[tuple[int, float], list[dict]] = {}
    order: list[tuple[int, float]] = []
    for det in dets:
        key = (det["shot_index"], det["t"])
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append(det)
    kept: list[dict] = []
    for key in order:
        group = groups[key]
        drop: set[int] = set()
        foods = [det for det in group if det["category"] == "food"]
        plates = [det for det in group if det["category"] == "plate"]
        for food in foods:
            for plate in plates:
                if not _overlaps_or_contains(food["box"], plate["box"]):
                    continue
                if _area(food["box"]) < _area(plate["box"]):
                    drop.add(id(food))
                elif _area(plate["box"]) < _area(food["box"]):
                    drop.add(id(plate))
                else:
                    drop.add(id(food))
        kept.extend(det for det in group if id(det) not in drop)
    return kept


def _matching_track(tracks: list[dict], det: dict) -> dict | None:
    for track in tracks:
        if track["shot_index"] != det["shot_index"] or track["category"] != det["category"]:
            continue
        if any(_overlaps_or_contains(det["box"], item["box"]) for item in track["items"]):
            return track
    return None


def _one_box_per_time(items: list[dict]) -> list[dict]:
    by_t: dict[float, dict] = {}
    for item in items:
        prev = by_t.get(item["t"])
        if prev is None or (item["visibility"], _area(item["box"])) > (
            prev["visibility"],
            _area(prev["box"]),
        ):
            by_t[item["t"]] = item
    return [by_t[t] for t in sorted(by_t)]


def _span(
    items: list[dict], shot: tuple[float, float], keyframes: list[float]
) -> tuple[float, float]:
    first = items[0]["t"]
    last = items[-1]["t"]
    start, end = first, last
    if keyframes:
        if abs(first - keyframes[0]) <= 1e-3:
            start = shot[0]
        if abs(last - keyframes[-1]) <= 1e-3:
            end = shot[1]
    if end < start:
        start, end = end, start
    return start, end


def _label_and_product(items: list[dict]) -> tuple[str, str]:
    groups: dict[str, list[dict]] = {}
    for item in items:
        groups.setdefault(item["label"], []).append(item)
    label = max(
        groups,
        key=lambda name: (len(groups[name]), max(item["visibility"] for item in groups[name])),
    )
    donor = max(groups[label], key=lambda item: item["visibility"])
    product = donor["suggested_product"]
    if not product:
        for item in sorted(items, key=lambda det: det["visibility"], reverse=True):
            if item["suggested_product"]:
                product = item["suggested_product"]
                break
    return label, product


if __name__ == "__main__":
    sys.exit(main())
