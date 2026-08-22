"""Run the audited Palisades Urban Wildfire GeoAI pipeline.

Usage:
    python run_pipeline.py
    python run_pipeline.py --from 30
    python run_pipeline.py --only 31
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

STAGES = [
    (1, "src/data/01_download_dins.py"),
    (2, "src/data/02_download_perimeter.py"),
    (10, "src/preprocessing/10_build_study_area.py"),
    (3, "src/data/03_download_gee_rasters.py"),
    (4, "src/data/04_download_landfire.py"),
    (5, "src/data/05_download_osm.py"),
    (6, "src/data/06_download_census.py"),
    (11, "src/preprocessing/11_qa_dins.py"),
    (12, "src/preprocessing/12_terrain_and_severity.py"),
    (20, "src/features/20_extract_predictors.py"),
    (30, "src/analysis/30_eda.py"),
    (31, "src/analysis/31_fit_models.py"),
    (32, "src/analysis/32_interpret.py"),
    (33, "src/analysis/33_severity_recovery.py"),
    (34, "src/analysis/34_community.py"),
    (40, "src/visualization/40_make_figures_maps.py"),
    (41, "src/visualization/41_make_figures_results.py"),
    (50, "src/manuscript/50_make_tables.py"),
    (51, "src/manuscript/51_manuscript_numbers.py"),
]
ORDER = [n for n, _ in STAGES]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", type=int, default=None)
    ap.add_argument("--only", type=int, default=None)
    args = ap.parse_args()
    stage_map = dict(STAGES)
    if args.only is not None:
        to_run = [args.only]
    elif args.start is not None:
        to_run = ORDER[ORDER.index(args.start):]
    else:
        to_run = list(ORDER)

    for n in to_run:
        script = stage_map[n]
        print(f"\n=== stage {n}: {script} ===")
        t0 = time.time()
        rc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT).returncode
        if rc != 0:
            print(f"FAILED at stage {n}. Resume with: python run_pipeline.py --from {n}")
            sys.exit(rc)
        print(f"    done in {time.time() - t0:.0f}s")
    print("\nPipeline complete. Optional: python -m pytest tests -q")


if __name__ == "__main__":
    main()
