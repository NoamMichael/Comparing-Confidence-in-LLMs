#!/usr/bin/env python3
"""Encode WGD photos to base64 data URIs for vision API calls.

Produces a Photos-encoded/ directory of .txt files (one per photo) and an
image_sizes.json manifest in Data/ mapping each photo stem to its encoded
file size in bytes.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

PHOTOS_DIR = Path(__file__).parent / "Data" / "Photos"
ENCODED_DIR = Path(__file__).parent / "Data" / "Photos-encoded"
MANIFEST = Path(__file__).parent / "Data" / "image_sizes.json"


def encode_all(
    photos_dir: Path = PHOTOS_DIR,
    encoded_dir: Path = ENCODED_DIR,
    manifest_path: Path = MANIFEST,
) -> int:
    encoded_dir.mkdir(parents=True, exist_ok=True)
    photos = [
        p for p in photos_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]
    sizes: dict[str, int] = {}
    encoded = 0
    for photo in photos:
        out = encoded_dir / f"{photo.stem}.txt"
        if not out.exists():
            raw = photo.read_bytes()
            mime = "image/png" if photo.suffix.lower() == ".png" else "image/jpeg"
            b64 = base64.b64encode(raw).decode()
            out.write_text(f"data:{mime};base64,{b64}")
            encoded += 1
        sizes[photo.stem] = out.stat().st_size
    manifest_path.write_text(json.dumps(sizes, sort_keys=True, indent=2))
    return encoded


def load_manifest(manifest_path: Path = MANIFEST) -> dict[str, int]:
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text())


def missing(photo_names: list[str], manifest_path: Path = MANIFEST) -> list[str]:
    manifest = load_manifest(manifest_path)
    return [
        name for name in sorted(set(photo_names))
        if Path(name).stem not in manifest
    ]


if __name__ == "__main__":
    n = encode_all()
    manifest = load_manifest()
    print(f"Encoded {n} new photos ({len(manifest)} total in {ENCODED_DIR})")
