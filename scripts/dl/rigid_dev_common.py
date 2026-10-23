"""Locked, subject-aware development sampling and comparable rigid metrics."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def available_cases(data_root: str | Path) -> list[dict]:
    root = Path(data_root) / "test"
    cases = []
    for d in sorted(root.glob("*/t*")):
        if not d.is_dir() or not (d / "params.json").is_file():
            continue
        p = json.loads((d / "params.json").read_text())
        group = d.parent.name
        subject = group.split("_rfMRI_", 1)[0]
        if subject == group:
            raise ValueError(f"Cannot resolve validation subject for {d}")
        cases.append({
            "relative_dir": str(d.relative_to(Path(data_root))),
            "subject": subject, "group": group,
            "orientation": p.get("orientation", "axial"),
            "thickness_mm": p.get("thickness_mm", p.get("thickness")),
            "position": p.get("position"),
            "misalignment": p.get("misalignment"),
        })
    if not cases:
        raise FileNotFoundError(f"No saved development samples in {root}")
    return cases


def select_cases(cases: list[dict], count: int = 36, seed: int = 2026) -> list[dict]:
    if not 1 <= count <= len(cases):
        raise ValueError("Count must be between 1 and number of available development cases")
    rng = np.random.default_rng(seed)
    strata = defaultdict(list)
    for case in cases:
        key = (case["orientation"], case["thickness_mm"], case["position"], case["misalignment"])
        strata[key].append(case)
    for group in strata.values():
        rng.shuffle(group)
    keys = sorted(strata, key=lambda x: tuple(str(v) for v in x))
    picked, usage = [], defaultdict(int)
    while len(picked) < count:
        progressed = False
        for key in keys:
            options = strata[key]
            if not options:
                continue
            i = min(range(len(options)), key=lambda idx: (usage[options[idx]["subject"]], idx))
            item = options.pop(i)
            picked.append(item)
            usage[item["subject"]] += 1
            progressed = True
            if len(picked) == count:
                break
        if not progressed:
            raise RuntimeError("Unable to select requested cases")
    return picked


def ensure_manifest(data_root: str | Path, path: str | Path, count: int = 36, seed: int = 2026) -> list[dict]:
    path = Path(path)
    if path.exists():
        saved = json.loads(path.read_text())
        if saved.get("data_split") != "validation" or not saved.get("cases"):
            raise ValueError(f"Invalid development manifest: {path}")
        cases = saved["cases"]
        missing = [c["relative_dir"] for c in cases if not (Path(data_root) / c["relative_dir"] / "slab.nii.gz").is_file()]
        if missing:
            raise FileNotFoundError(f"Missing validation samples: {missing[:3]}")
        return cases
    cases = select_cases(available_cases(data_root), count=count, seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_split": "validation", "seed": seed, "cases": cases}
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2) + "\n")
    temp.replace(path)
    return cases


def assert_rigid(matrix: np.ndarray, *, tolerance: float = 2e-2) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (4, 4) or not np.isfinite(m).all():
        raise ValueError("Expected finite 4x4 transform")
    if not np.allclose(m[3], [0, 0, 0, 1], atol=1e-6):
        raise ValueError("Invalid homogeneous transform bottom row")
    if not np.allclose(m[:3, :3].T @ m[:3, :3], np.eye(3), atol=tolerance):
        raise ValueError("Transform contains scale or shear")
    if not np.isclose(np.linalg.det(m[:3, :3]), 1.0, atol=tolerance):
        raise ValueError("Transform is not a proper rigid rotation")
    return m


def load_rows(path: str | Path) -> dict[tuple[str, str], dict]:
    out = {}
    p = Path(path)
    if p.is_file():
        for line in p.read_text().splitlines():
            if line.strip():
                record = json.loads(line)
                out[(record["case"], record["method"])] = record
    return out


def append_row(path: str | Path, record: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as file:
        file.write(json.dumps(record, allow_nan=False) + "\n")
        file.flush()


def summarize(rows: list[dict]) -> dict:
    summary = {}
    for method in sorted(set(r["method"] for r in rows)):
        subset = [r for r in rows if r["method"] == method]
        valid = [r for r in subset if r.get("status") == "ok"]
        errors = [r.get("mean_tre_mm") for r in valid]
        failures = [r.get("failure", True) for r in subset]
        summary[method] = {
            "attempted": len(subset), "succeeded": len(valid),
            "tool_failure_pct": 100 * (len(subset) - len(valid)) / len(subset),
            "failure_pct": 100 * float(np.mean(failures)),
            "median_tre_mm": float(np.median(errors)) if errors else None,
            "median_rotation_error_deg": float(np.median([r["rot_deg"] for r in valid])) if valid else None,
            "median_centroid_translation_error_mm": float(np.median([r["trans_mm"] for r in valid])) if valid else None,
            "tre_below_2mm_pct": 100 * sum(r["mean_tre_mm"] < 2 for r in valid) / len(subset),
            "median_runtime_s": float(np.median([r["runtime_s"] for r in valid])) if valid else None,
        }
    return summary


def print_summary(rows: list[dict], output: str | Path) -> dict:
    result = summarize(rows)
    Path(output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    return result
