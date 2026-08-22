"""Core project utilities: configuration, paths, logging, and the data manifest.

Every pipeline script imports from this module so that paths, CRS, and
parameters come from ``config/project.yaml`` rather than being hard-coded.
"""
from __future__ import annotations

import csv
import hashlib
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import yaml

# Repository root = two levels above this file (src/utils/project.py)
ROOT = Path(__file__).resolve().parents[2]


def load_config() -> dict:
    """Load the central project configuration."""
    with open(ROOT / "config" / "project.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def path(*parts: str) -> Path:
    """Resolve a path relative to the repository root, creating parents."""
    p = ROOT.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    """Logger writing to console and outputs/logs/pipeline.log."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(name)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    logdir = ROOT / "outputs" / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(logdir / "pipeline.log", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    return logger


def sha256(fp: Path, chunk: int = 1 << 20) -> str:
    """SHA-256 checksum of a file."""
    h = hashlib.sha256()
    with open(fp, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


MANIFEST_FIELDS = [
    "dataset_id", "dataset_name", "provider", "landing_page", "access_url",
    "access_method", "license", "temporal_coverage", "spatial_coverage",
    "resolution", "native_crs", "variables", "analytical_role",
    "retrieved_utc", "local_path", "file_bytes", "sha256", "version",
    "limitations", "credentials_required",
]


def record_manifest(entry: dict) -> None:
    """Append/update a row in data/data_manifest.csv (keyed by dataset_id + local_path)."""
    cfg = load_config()
    mpath = ROOT / cfg["paths"]["manifest"]
    mpath.parent.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    if mpath.exists():
        with open(mpath, "r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))

    key = (entry.get("dataset_id"), entry.get("local_path"))
    rows = [r for r in rows if (r.get("dataset_id"), r.get("local_path")) != key]
    full = {k: str(entry.get(k, "")) for k in MANIFEST_FIELDS}
    rows.append(full)
    rows.sort(key=lambda r: r["dataset_id"])

    with open(mpath, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)


def utcnow() -> str:
    """Current UTC timestamp string."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_info(fp: Path) -> dict:
    """Byte size and checksum for manifest records."""
    return {"file_bytes": fp.stat().st_size, "sha256": sha256(fp)}
