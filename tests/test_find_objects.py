"""Box conversion, per-shot merge, and the one-keyframe drop rule. HTTP is mocked."""

from __future__ import annotations

import base64
import io
import json
import re
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from find_objects import (
    FRAMES_PER_CALL,
    MODEL,
    VISIBILITY_MIN,
    _post_openrouter,
    box_2d_to_xywh,
    find_objects,
    keyframe_times,
    merge_objects,
    prompt_for,
    scene_cuts,
    shot_ranges,
)

_KEY = "test-openrouter-key"


def _box(x, y, w, h):
    return {"x": x, "y": y, "w": w, "h": h}


def _det(**overrides):
    raw = {
        "shot_index": 1,
        "t": 2.15,
        "label": "green wine bottle",
        "category": "bottle",
        "suggested_product": "branded wine bottle",
        "visibility": 0.85,
        "box": _box(100, 40, 40, 90),
    }
    raw.update(overrides)
    return raw


def test_box_2d_is_ymin_xmin_ymax_xmax_on_a_0_to_1000_scale():
    box = box_2d_to_xywh([0, 500, 1000, 1000], 1000, 1000)
    assert box == {"x": 500, "y": 0, "w": 500, "h": 1000}
    full = box_2d_to_xywh([0, 0, 1000, 1000], 320, 180)
    assert full == {"x": 0, "y": 0, "w": 320, "h": 180}
    clamped = box_2d_to_xywh([-20, -10, 1200, 1500], 200, 100)
    assert clamped == {"x": 0, "y": 0, "w": 200, "h": 100}
    with pytest.raises(ValueError):
        box_2d_to_xywh([100, 100, 100, 400], 100, 100)
    # 0-1 fractions are not treated as a fraction of the frame.
    tiny = box_2d_to_xywh([0.2, 0.1, 0.8, 0.4], 1000, 1000)
    assert tiny == {"x": 0, "y": 0, "w": 1, "h": 1}


def test_keyframe_times_are_about_one_per_second_and_clear_of_cuts():
    times = keyframe_times(0.0, 10.0)
    assert len(times) == 10
    assert times[0] >= 0.15
    assert times[-1] <= 10.0 - 0.15
    gaps = [b - a for a, b in zip(times, times[1:])]
    assert all(0.5 <= gap <= 1.5 for gap in gaps)
    short = keyframe_times(2.0, 4.0)
    assert len(short) == 3
    assert short[0] >= 2.15
    assert short[-1] <= 3.85
    # Exactly 0.3s can sit one frame on the margin. Anything shorter is a cut frame.
    assert keyframe_times(0.0, 0.3) == [0.15]
    assert keyframe_times(1.0, 1.2) == []


def test_shot_ranges_are_the_spans_between_cuts():
    assert shot_ranges([2.0, 4.0], 10.0) == [(0.0, 2.0), (2.0, 4.0), (4.0, 10.0)]
    assert shot_ranges([], 10.0) == [(0.0, 10.0)]
    assert shot_ranges([0.0, 10.0, 4.0, 4.0], 10.0) == [(0.0, 4.0), (4.0, 10.0)]


def test_merge_overlaps_inside_a_shot_and_extends_to_the_shot_edge():
    """The Suits dinner bottle is seen on the first and last keyframe, so it fills 2–4s."""
    frames = [2.15, 3.0, 3.85]
    bottle_boxes = (
        (2.15, _box(100, 40, 40, 90), "green wine bottle"),
        (3.0, _box(102, 42, 38, 88), "wine bottle"),
        (3.85, _box(101, 41, 39, 89), "green wine bottle"),
    )
    glass_a = (_box(10, 50, 30, 70), "wine glass")
    glass_b = (_box(220, 50, 30, 70), "second wine glass")
    dets = []
    for t, box, label in bottle_boxes:
        dets.append(_det(t=t, box=box, label=label))
    for t in frames:
        dets.append(
            _det(
                t=t,
                label=glass_a[1],
                category="glass",
                suggested_product="branded glass",
                visibility=0.8,
                box=glass_a[0],
            )
        )
        dets.append(
            _det(
                t=t,
                label=glass_b[1],
                category="glass",
                suggested_product="branded glass",
                visibility=0.7,
                box=glass_b[0],
            )
        )
    dets.append(
        _det(
            t=3.0,
            label="wristwatch",
            category="device",
            suggested_product="luxury watch",
            visibility=0.45,
            box=_box(70, 30, 18, 12),
        )
    )
    dets.append(
        _det(
            t=2.15,
            label="waiter bill folder",
            category="packaging",
            suggested_product="branded folder",
            visibility=0.25,
            box=_box(300, 20, 40, 50),
        )
    )
    # Same pixels in another shot stay a different object.
    dets.append(_det(shot_index=0, t=0.15, box=bottle_boxes[0][1]))
    dets.append(_det(shot_index=0, t=1.85, box=bottle_boxes[0][1], label="wine bottle"))

    found = merge_objects(
        dets,
        [(0.0, 2.0), (2.0, 4.0)],
        [[0.15, 1.0, 1.85], frames],
    )
    by_key = {(obj["shot_index"], obj["label"]): obj for obj in found}
    assert "waiter bill folder" not in {obj["label"] for obj in found}
    bottle = by_key[(1, "green wine bottle")]
    assert bottle["start_s"] == 2.0
    assert bottle["end_s"] == 4.0
    assert bottle["category"] == "bottle"
    assert bottle["suggested_product"] == "branded wine bottle"
    assert bottle["visibility"] == pytest.approx(0.85)
    assert [b["t"] for b in bottle["boxes"]] == frames
    assert bottle["boxes"][0] == {"t": 2.15, **bottle_boxes[0][1]}
    assert by_key[(1, "wine glass")]["start_s"] == 2.0
    assert by_key[(1, "second wine glass")]["end_s"] == 4.0
    assert by_key[(1, "wine glass")]["boxes"][0]["x"] != by_key[(1, "second wine glass")]["boxes"][0]["x"]
    watch = by_key[(1, "wristwatch")]
    assert watch["visibility"] >= VISIBILITY_MIN
    assert watch["start_s"] == watch["end_s"] == 3.0
    other = by_key[(0, "green wine bottle")]
    assert (other["start_s"], other["end_s"]) == (0.0, 2.0)
    assert len(other["boxes"]) == 2


def test_contained_box_merges_when_iou_is_small():
    big = _box(0, 0, 200, 200)
    small = _box(90, 90, 20, 20)
    inter = 20 * 20
    union = 200 * 200 + 20 * 20 - inter
    assert inter / union < 0.05
    found = merge_objects(
        [
            _det(shot_index=0, t=1.0, box=big, label="loose bottle"),
            _det(shot_index=0, t=2.0, box=small, label="tight bottle", visibility=0.95),
        ],
        [(0.0, 3.0)],
        [[0.5, 1.0, 2.0, 2.5]],
    )
    assert len(found) == 1
    assert found[0]["start_s"] == 1.0
    assert found[0]["end_s"] == 2.0
    assert [b["t"] for b in found[0]["boxes"]] == [1.0, 2.0]


def test_food_and_plate_overlap_keeps_the_larger_box():
    plate = _box(50, 80, 120, 40)
    steak = _box(70, 88, 40, 20)
    shots = [(0.0, 4.0)]
    frames = [[0.15, 2.0, 3.85]]
    food_first = merge_objects(
        [
            _det(shot_index=0, t=2.0, label="steak", category="food", visibility=0.95, box=steak),
            _det(shot_index=0, t=2.0, label="dinner plate", category="plate", visibility=0.7, box=plate),
        ],
        shots,
        frames,
    )
    assert [obj["label"] for obj in food_first] == ["dinner plate"]
    assert food_first[0]["boxes"] == [{"t": 2.0, **plate}]

    larger_food = merge_objects(
        [
            _det(shot_index=0, t=2.0, label="steak", category="food", visibility=0.9, box=plate),
            _det(shot_index=0, t=2.0, label="saucer", category="plate", visibility=0.8, box=steak),
        ],
        shots,
        frames,
    )
    assert [obj["label"] for obj in larger_food] == ["steak"]

    apart = merge_objects(
        [
            _det(shot_index=0, t=2.0, label="steak", category="food", visibility=0.9, box=_box(0, 0, 20, 20)),
            _det(shot_index=0, t=2.0, label="dinner plate", category="plate", visibility=0.7, box=_box(80, 80, 40, 20)),
        ],
        shots,
        frames,
    )
    assert {obj["label"] for obj in apart} == {"steak", "dinner plate"}


def test_drop_rule_is_one_keyframe_below_0_3():
    shots = [(0.0, 4.0)]
    frames = [[0.15, 2.0, 3.85]]
    found = merge_objects(
        [
            _det(shot_index=0, t=2.0, label="dim can", category="can", visibility=0.29, box=_box(1, 1, 10, 10)),
            _det(shot_index=0, t=2.0, label="clear can", category="can", visibility=0.3, box=_box(40, 1, 10, 10)),
            _det(shot_index=0, t=0.15, label="faint bottle", category="bottle", visibility=0.1, box=_box(1, 30, 10, 20)),
            _det(shot_index=0, t=3.85, label="faint bottle", category="bottle", visibility=0.1, box=_box(2, 31, 10, 20)),
        ],
        shots,
        frames,
    )
    labels = {obj["label"] for obj in found}
    assert "dim can" not in labels
    assert "clear can" in labels
    faint = next(obj for obj in found if obj["label"] == "faint bottle")
    assert faint["visibility"] == pytest.approx(0.1)
    assert (faint["start_s"], faint["end_s"]) == (0.0, 4.0)
    assert len(faint["boxes"]) == 2


def _write_clip(path: Path) -> None:
    """4s red then 6s blue, so the scene cut sits at 4s."""
    cmd = ["ffmpeg", "-y", "-hide_banner", "-loglevel", "error"]
    for color, dur in (("red", 4), ("blue", 6)):
        cmd += ["-f", "lavfi", "-i", f"color=c={color}:s=320x180:d={dur}:r=25"]
    cmd += [
        "-filter_complex",
        "[0:v][1:v]concat=n=2:v=1:a=0",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(path),
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def _rgb_mean(jpeg: bytes) -> tuple[float, float, float]:
    proc = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"],
        input=jpeg,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    raw = proc.stdout
    n = len(raw) // 3
    return (
        sum(raw[0::3]) / n,
        sum(raw[1::3]) / n,
        sum(raw[2::3]) / n,
    )


def _jpeg_size(data: bytes) -> tuple[int, int]:
    from find_objects import _jpeg_size as size

    return size(data)


def test_find_objects_converts_merges_and_drops_with_http_mocked(tmp_path, monkeypatch):
    from find_objects import probe_video

    monkeypatch.setenv("OPENROUTER_API_KEY", _KEY)
    video = tmp_path / "clip.mp4"
    _write_clip(video)
    cuts = scene_cuts(video)
    assert len(cuts) == 1
    assert cuts[0] == pytest.approx(4.0, abs=0.15)
    duration, width, height = probe_video(video)
    assert (width, height) == (320, 180)
    shots = shot_ranges(cuts, duration)
    assert len(shots) == 2
    sampled = [keyframe_times(start, end) for start, end in shots]
    assert sum(len(times) for times in sampled) > FRAMES_PER_CALL
    phone_t = sampled[0][1]
    plate_t = sampled[1][len(sampled[1]) // 2]
    assert phone_t not in (sampled[0][0], sampled[0][-1])
    assert plate_t not in (sampled[1][0], sampled[1][-1])

    calls = []

    def request(payload):
        calls.append(payload)
        content = payload["messages"][0]["content"]
        text = content[0]["text"]
        images = [part for part in content if part["type"] == "image_url"]
        assert _KEY not in json.dumps({"model": payload["model"], "text": text})
        objects = []
        listed = [
            (int(i), float(t)) for i, t in re.findall(r"Frame (\d+) at ([0-9.]+)s", text)
        ]
        assert [i for i, _ in listed] == list(range(len(images)))
        for index, t in listed:
            shot = 0 if t < cuts[0] else 1
            if shot == 1:
                objects.append(
                    {
                        "frame": index,
                        "label": "wine bottle" if abs(t - sampled[1][1]) < 0.02 else "green wine bottle",
                        "category": "bottle",
                        "suggested_product": "branded wine bottle",
                        "visibility": 0.9,
                        "box_2d": [100, 100 + index, 400, 180 + index],
                    }
                )
                objects.append(
                    {
                        "frame": index,
                        "label": "wine glass",
                        "category": "glass",
                        "suggested_product": "branded glass",
                        "visibility": 0.8,
                        "box_2d": [100, 400, 500, 520],
                    }
                )
                objects.append(
                    {
                        "frame": index,
                        "label": "second wine glass",
                        "category": "glass",
                        "suggested_product": "branded glass",
                        "visibility": 0.75,
                        "box_2d": [100, 700, 500, 900],
                    }
                )
            if abs(t - phone_t) < 0.02:
                objects.append(
                    {
                        "frame": index,
                        "label": "black phone",
                        "category": "device",
                        "suggested_product": "branded phone",
                        "visibility": 0.6,
                        "box_2d": [100, 100, 280, 250],
                    }
                )
            if abs(t - sampled[0][0]) < 0.02:
                objects.append(
                    {
                        "frame": index,
                        "label": "coffee mug",
                        "category": "cup",
                        "suggested_product": "branded mug",
                        "visibility": 0.2,
                        "box_2d": [10, 10, 80, 80],
                    }
                )
                objects.append(
                    {
                        "frame": index,
                        "label": "empty table",
                        "category": "surface",
                        "suggested_product": "nothing",
                        "visibility": 0.99,
                        "box_2d": [500, 0, 1000, 1000],
                    }
                )
            if abs(t - plate_t) < 0.02:
                objects.append(
                    {
                        "frame": index,
                        "label": "steak",
                        "category": "food",
                        "suggested_product": "packaged meal",
                        "visibility": 0.95,
                        "box_2d": [700, 300, 800, 500],
                    }
                )
                objects.append(
                    {
                        "frame": index,
                        "label": "dinner plate",
                        "category": "plate",
                        "suggested_product": "branded plate",
                        "visibility": 0.7,
                        "box_2d": [600, 200, 900, 700],
                    }
                )
                objects.append(
                    {
                        "frame": index,
                        "label": "bad box",
                        "category": "can",
                        "suggested_product": "soda",
                        "visibility": 0.9,
                        "box_2d": [10, 10, 10, 40],
                    }
                )
        return {
            "choices": [{"message": {"content": json.dumps({"objects": objects})}}],
            "usage": {"cost": 0.0125},
        }

    usage = {}
    found = find_objects(video, request=request, usage=usage)
    assert calls
    assert [len([p for p in call["messages"][0]["content"] if p["type"] == "image_url"]) for call in calls] == [
        FRAMES_PER_CALL,
        sum(len(times) for times in sampled) - FRAMES_PER_CALL,
    ]
    for call in calls:
        assert call["model"] == MODEL
        assert call["reasoning"] == {"effort": "low"}
        assert call["response_format"] == {"type": "json_object"}
        text = call["messages"][0]["content"][0]["text"]
        listed_times = [float(t) for _, t in re.findall(r"Frame (\d+) at ([0-9.]+)s", text)]
        assert text == prompt_for(listed_times)
        assert "EXISTING visible objects" in text
        assert "[ymin, xmin, ymax, xmax]" in text
        assert "0-1000" in text
        assert "contact sheet" not in text.lower()
        for part in call["messages"][0]["content"][1:]:
            url = part["image_url"]["url"]
            assert url.startswith("data:image/jpeg;base64,")
            raw = base64.b64decode(url.split(",", 1)[1])
            assert _jpeg_size(raw) == (320, 180)

    colors = []
    for call in calls:
        for part in call["messages"][0]["content"]:
            if part["type"] != "image_url":
                continue
            raw = base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
            colors.append(_rgb_mean(raw))
    assert len(colors) == sum(len(times) for times in sampled)
    reds, blues = colors[: len(sampled[0])], colors[len(sampled[0]) :]
    assert all(r > g + 40 and r > b + 40 for r, g, b in reds)
    assert all(b > r + 40 and b > g + 40 for r, g, b in blues)

    by_label = {obj["label"]: obj for obj in found}
    assert "coffee mug" not in by_label
    assert "empty table" not in by_label
    assert "steak" not in by_label
    assert "bad box" not in by_label
    bottle = by_label["green wine bottle"]
    assert bottle["shot_index"] == 1
    assert bottle["start_s"] == pytest.approx(shots[1][0], abs=0.01)
    assert bottle["end_s"] == pytest.approx(shots[1][1], abs=0.01)
    assert len(bottle["boxes"]) == len(sampled[1])
    first_bottle = box_2d_to_xywh([100, 100 + len(sampled[0]), 400, 180 + len(sampled[0])], 320, 180)
    assert {k: bottle["boxes"][0][k] for k in ("x", "y", "w", "h")} == first_bottle
    assert bottle["suggested_product"] == "branded wine bottle"
    assert by_label["wine glass"]["shot_index"] == 1
    assert by_label["second wine glass"]["shot_index"] == 1
    phone = by_label["black phone"]
    assert phone["start_s"] == phone["end_s"] == pytest.approx(phone_t, abs=0.02)
    assert phone["boxes"] == [
        {"t": phone["boxes"][0]["t"], **box_2d_to_xywh([100, 100, 280, 250], 320, 180)}
    ]
    plate = by_label["dinner plate"]
    assert len(plate["boxes"]) == 1
    assert plate["boxes"][0]["w"] > 20
    assert usage["calls"] == 2
    assert usage["cost_usd"] == pytest.approx(0.025)


def test_prompt_names_the_fields_and_the_frame_times():
    text = prompt_for([0.15, 1.0])
    assert "Frame 0 at 0.150s" in text
    assert "Frame 1 at 1.000s" in text
    assert "suggested_product" in text
    assert "visibility" in text
    assert "category food" in text
    assert "plate" in text


def test_ask_reads_frames_retries_and_redacts(tmp_path, capsys):
    jpeg = tmp_path / "f.jpg"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=red:s=16x16:d=0.04",
            "-frames:v",
            "1",
            str(jpeg),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from find_objects import _ask

    calls = {"n": 0}

    def request(payload):
        calls["n"] += 1
        assert payload["model"] == MODEL
        assert payload["reasoning"] == {"effort": "low"}
        images = [p for p in payload["messages"][0]["content"] if p["type"] == "image_url"]
        assert len(images) == 1
        assert _KEY not in json.dumps(payload["messages"][0]["content"][0])
        if calls["n"] == 1:
            raise RuntimeError(f"bad json {_KEY}")
        body = [{"type": "text", "text": json.dumps({"objects": [{"frame": 0, "label": "can"}]})}]
        return {"choices": [{"message": {"content": body}}]}

    items = _ask([{"t": 1.0, "path": jpeg}], request, _KEY)
    assert calls["n"] == 2
    assert items == [{"frame": 0, "label": "can"}]
    assert capsys.readouterr().out == ""


def test_http_error_redacts_the_key(monkeypatch):
    def urlopen(req, timeout=0):
        raise urllib.error.HTTPError(
            req.full_url,
            401,
            "unauthorized",
            hdrs=None,
            fp=io.BytesIO(f"nope {_KEY}".encode()),
        )

    monkeypatch.setattr(urllib.request, "urlopen", urlopen)
    with pytest.raises(RuntimeError) as exc:
        _post_openrouter({"model": MODEL}, _KEY)
    assert _KEY not in str(exc.value)
    assert "[redacted]" in str(exc.value)
    sent = {}

    class Resp:
        def read(self):
            return b'{"choices":[{"message":{"content":"{}"}}]}'

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def ok(req, timeout=0):
        sent["auth"] = req.get_header("Authorization")
        sent["body"] = json.loads(req.data.decode())
        return Resp()

    monkeypatch.setattr(urllib.request, "urlopen", ok)
    _post_openrouter({"model": MODEL, "messages": []}, _KEY)
    assert sent["auth"] == f"Bearer {_KEY}"
    assert _KEY not in json.dumps(sent["body"])


def test_find_objects_requires_a_key(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        find_objects(tmp_path / "missing.mp4")


def test_duration_follows_the_video_stream_not_a_longer_audio_track(tmp_path):
    """A 1s picture with 3s of audio used to be probed as 3s, then frame 1.5s failed."""
    video = tmp_path / "long-audio.mp4"
    subprocess.check_call(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "lavfi",
            "-i",
            "color=c=green:s=64x64:d=1:r=10",
            "-f",
            "lavfi",
            "-i",
            "sine=d=3",
            "-c:v",
            "libx264",
            "-c:a",
            "aac",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    from find_objects import probe_video

    duration, width, height = probe_video(video)
    assert (width, height) == (64, 64)
    assert duration == pytest.approx(1.0, abs=0.05)
    calls = {"n": 0}

    def request(payload):
        calls["n"] += 1
        return {"choices": [{"message": {"content": '{"objects":[]}'}}], "usage": {"cost": 0.001}}

    usage = {}
    found = find_objects(video, request=request, usage=usage)
    assert found == []
    assert calls["n"] == 1
    assert usage["cost_usd"] == pytest.approx(0.001)
    times = keyframe_times(0.0, duration)
    assert times
    assert max(times) < duration


def test_one_bad_batch_keeps_objects_from_the_batches_that_worked(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video)
    calls = {"n": 0}

    def request(payload):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("bad json")
        text = payload["messages"][0]["content"][0]["text"]
        objects = []
        for index, t in re.findall(r"Frame (\d+) at ([0-9.]+)s", text):
            objects.append(
                {
                    "frame": int(index),
                    "label": "green wine bottle",
                    "category": "bottle",
                    "suggested_product": "branded wine bottle",
                    "visibility": 0.9,
                    "box_2d": [100, 100, 400, 200],
                }
            )
        return {"choices": [{"message": {"content": json.dumps({"objects": objects})}}]}

    found = find_objects(video, request=request)
    assert calls["n"] == 3
    assert any(obj["label"] == "green wine bottle" for obj in found)


def test_every_batch_failing_raises(tmp_path):
    video = tmp_path / "clip.mp4"
    _write_clip(video)

    def request(payload):
        raise RuntimeError("bad json")

    with pytest.raises(RuntimeError, match="no frame batch"):
        find_objects(video, request=request)


def test_reasoning_preamble_does_not_drop_the_json():
    from find_objects import _message_text, _object_items

    body = {
        "choices": [
            {
                "message": {
                    "content": [
                        {"type": "reasoning", "text": "The bottle is on the table."},
                        {
                            "type": "text",
                            "text": 'Note:\n{"objects":[{"frame":0,"label":"wine bottle"}]}',
                        },
                    ]
                }
            }
        ]
    }
    assert _object_items(_message_text(body)) == [{"frame": 0, "label": "wine bottle"}]


def test_preamble_box_does_not_replace_the_objects():
    """A box_2d array before the payload must not be treated as the batch result."""
    from find_objects import _object_items

    text = (
        "Frames [0, 1] and box_2d [100, 200, 300, 400].\n"
        '{"objects":[{"frame":0,"label":"wine bottle"}]}'
    )
    assert _object_items(text) == [{"frame": 0, "label": "wine bottle"}]
    with pytest.raises(RuntimeError, match="missing objects"):
        _object_items("[100, 200, 300, 400]")


def test_save_flag_stores_the_clip_for_the_ui(tmp_path, monkeypatch, capsys):
    from find_objects import main
    from store import list_clips

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    video = tmp_path / "suits.mp4"
    video.write_bytes(b"video")

    def fake(path, usage=None, request=None):
        usage["cost_usd"] = 0.025
        usage["calls"] = 1
        return [
            {
                "label": "green wine bottle",
                "category": "bottle",
                "suggested_product": "branded wine bottle",
                "shot_index": 0,
                "start_s": 2.0,
                "end_s": 4.0,
                "visibility": 0.9,
                "boxes": [{"t": 2.15, "x": 10, "y": 20, "w": 30, "h": 40}],
            }
        ]

    monkeypatch.setattr("find_objects.find_objects", fake)
    assert main(["--save", str(video)]) == 0
    captured = capsys.readouterr()
    saved = json.loads(captured.out)
    assert saved[0]["label"] == "green wine bottle"
    assert "cost_usd 0.025000" in captured.err
    assert _KEY not in captured.err
    clips = list_clips()
    assert clips[0]["name"] == "suits.mp4"
    assert clips[0]["cost_usd"] == 0.025
    assert clips[0]["objects"][0]["start_s"] == 2.0
    assert clips[0]["objects"][0]["end_s"] == 4.0
    assert clips[0]["objects"][0]["boxes"][0]["x"] == 10
