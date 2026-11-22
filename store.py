"""Local catalog of finder results. The UI reads this same file.

``DATA_DIR`` overrides the directory. Unset, results live in ``runs/``.
"""

from __future__ import annotations

import json
import os
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def catalog_dir(root: Path | None = None) -> Path:
    if root is not None:
        path = Path(root)
    else:
        raw = (os.environ.get("DATA_DIR") or "").strip()
        path = Path(raw) if raw else Path("runs")
    path.mkdir(parents=True, exist_ok=True)
    return path


def db_path(root: Path | None = None) -> Path:
    return catalog_dir(root) / "finder.sqlite"


def connect(root: Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(root))
    conn.row_factory = sqlite3.Row
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS clips (
            id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            video_path TEXT NOT NULL,
            cost_usd REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS objects (
            id TEXT PRIMARY KEY,
            clip_id TEXT NOT NULL,
            label TEXT NOT NULL,
            category TEXT,
            suggested_product TEXT,
            start_s REAL NOT NULL,
            end_s REAL NOT NULL,
            visibility REAL,
            boxes_json TEXT NOT NULL
        )
        """
    )
    conn.commit()
    return conn


def save_findings(
    video_path: str | Path,
    objects: list[dict],
    cost_usd: float | None = None,
    *,
    root: Path | None = None,
) -> str:
    """Replace any stored result for this video and return the clip id."""
    path = Path(video_path).resolve()
    now = datetime.now(timezone.utc).isoformat()
    conn = connect(root)
    try:
        row = conn.execute("SELECT id FROM clips WHERE video_path = ?", (str(path),)).fetchone()
        clip_id = row["id"] if row else uuid.uuid4().hex[:12]
        if row:
            conn.execute("DELETE FROM objects WHERE clip_id = ?", (clip_id,))
            conn.execute(
                "UPDATE clips SET name = ?, cost_usd = ?, created_at = ? WHERE id = ?",
                (path.name, cost_usd, now, clip_id),
            )
        else:
            conn.execute(
                "INSERT INTO clips (id, name, video_path, cost_usd, created_at) VALUES (?, ?, ?, ?, ?)",
                (clip_id, path.name, str(path), cost_usd, now),
            )
        for obj in objects:
            conn.execute(
                """
                INSERT INTO objects (
                    id, clip_id, label, category, suggested_product,
                    start_s, end_s, visibility, boxes_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    uuid.uuid4().hex[:12],
                    clip_id,
                    str(obj.get("label") or "object"),
                    str(obj.get("category") or ""),
                    str(obj.get("suggested_product") or ""),
                    float(obj["start_s"]),
                    float(obj["end_s"]),
                    float(obj.get("visibility") or 0),
                    json.dumps(obj.get("boxes") or []),
                ),
            )
        conn.commit()
        return clip_id
    finally:
        conn.close()


def list_clips(root: Path | None = None) -> list[dict[str, Any]]:
    conn = connect(root)
    try:
        clips = conn.execute("SELECT * FROM clips ORDER BY created_at, name").fetchall()
        return [_clip_dict(conn, clip) for clip in clips]
    finally:
        conn.close()


def get_clip(clip_id: str, root: Path | None = None) -> dict[str, Any] | None:
    conn = connect(root)
    try:
        clip = conn.execute("SELECT * FROM clips WHERE id = ?", (clip_id,)).fetchone()
        if clip is None:
            return None
        return _clip_dict(conn, clip)
    finally:
        conn.close()


def _clip_dict(conn: sqlite3.Connection, clip: sqlite3.Row) -> dict[str, Any]:
    rows = conn.execute(
        "SELECT * FROM objects WHERE clip_id = ? ORDER BY start_s, label",
        (clip["id"],),
    ).fetchall()
    objects = []
    for row in rows:
        objects.append(
            {
                "id": row["id"],
                "label": row["label"],
                "category": row["category"],
                "suggested_product": row["suggested_product"],
                "start_s": row["start_s"],
                "end_s": row["end_s"],
                "visibility": row["visibility"],
                "boxes": json.loads(row["boxes_json"]),
            }
        )
    return {
        "id": clip["id"],
        "name": clip["name"],
        "video_path": clip["video_path"],
        "cost_usd": clip["cost_usd"],
        "objects": objects,
    }
