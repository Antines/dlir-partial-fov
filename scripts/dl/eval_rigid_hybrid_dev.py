#!/usr/bin/env python3
"""Locked development comparison: header, trained CNN, and gated 6-DOF refinement."""
from __future__ import annotations

import argparse
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from dl.rigid_dev_common import append_row, assert_rigid, ensure_manifest, load_rows, print_summary
from dl.rigid_dev_wandb import start_tracking
from dl.rigid_data import SavedSlabDataset
from dl.rigid_eval import score_matrix
from dl.rigid_refine import RefinementConfig, refine_rigid, score_rigid_similarity
from slabsim.evaluation import fit_rigid
from slabsim.geometry import voxel_centers


def load_spatial_fusion(path: Path, device: torch.device):
    from dl.rigid_model_spatial_fusion_v1 import RigidRegistrationNet
    state = torch.load(path, map_location="cpu", weights_only=True)
    cfg = state["config"]["model"]
    network = RigidRegistrationNet(
        grid_shape_xyz=cfg["grid_shape_xyz"], base_channels=cfg["base_channels"],
        max_rotation_deg=cfg["max_rotation_deg"], max_translation_mm=cfg["max_translation_mm"],
    ).to(device)
    network.load_state_dict(state["model"], strict=True)
    return network.eval()


@lru_cache(maxsize=6)
def load_mask(path: str) -> tuple[torch.Tensor, torch.Tensor]:
    image = nib.load(path)
    return (torch.from_numpy(image.get_fdata(dtype=np.float32)[None]),
            torch.from_numpy(np.asarray(image.affine, dtype=np.float32)))


def score_candidate(matrix: torch.Tensor, sample: dict, *, method: str, elapsed: float, relative: str, subject: str) -> dict:
    mat = assert_rigid(matrix.detach().cpu().numpy())
    m = score_matrix(mat, sample)
    return {
        "case": relative, "subject": subject, "method": method, "status": "ok",
        "mean_tre_mm": m["mean"], "rot_deg": m["rot_deg"], "trans_mm": m["trans_mm"],
        "failure": m["failure"], "runtime_s": elapsed,
        "rotation_magnitude_deg": float(np.degrees(np.arccos(np.clip((np.trace(mat[:3, :3]) - 1) / 2, -1, 1)))),
    }


def args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--max-cases", type=int, default=36)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--config", type=Path, default=REPO / "configs/rigid_refine.yaml")
    p.add_argument("--mode", choices=["score", "refine"], default="score")
    p.add_argument("--score-summary", type=Path)
    p.add_argument("--min-oracle-preference", type=float, default=0.75)
    p.add_argument("--wandb-project", default="neuroml-dlir")
    return p.parse_args()


def main():
    a = args()
    a.output.mkdir(parents=True, exist_ok=True)
    cases = ensure_manifest(a.data, a.manifest, count=a.max_cases)
    config_data = yaml.safe_load(a.config.read_text())["refinement"]
    config = RefinementConfig(**config_data)
    if a.mode == "refine":
        if a.score_summary is None or not a.score_summary.is_file():
            raise ValueError("Refinement requires a completed --score-summary from the same manifest")
        data = json.loads(a.score_summary.read_text())
        expected = [c["relative_dir"] for c in cases]
        if (data.get("case_ids") != expected or data.get("similarity") != config.similarity
                or data.get("oracle_preferred_pct", 0) < 100 * a.min_oracle_preference):
            raise RuntimeError("The locked objective-ranking gate did not pass; no refinement will run")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"{a.mode} evaluation: {len(cases)} locked cases, device={device}", flush=True)
    if a.checkpoint is not None:
        print("Loading archived spatial-fusion checkpoint", flush=True)
    model = load_spatial_fusion(a.checkpoint, device) if a.checkpoint is not None else None
    if model is None and a.mode == "refine":
        print("No CNN checkpoint supplied; evaluating header-initialized refinement only")
    run = start_tracking(a.output, project=a.wandb_project,
                         name=(f"rigid-header-{a.mode}-val1080-v1" if len(cases) == 1080 and model is None else f"rigid-hybrid-{a.mode}-dev-v1"), config={"mode": a.mode, "similarity": config.similarity, "cases": len(cases), "cnn_checkpoint": str(a.checkpoint) if a.checkpoint else None})
    done = load_rows(a.output / "cases.jsonl")
    oracle_votes = []
    objectives = []
    try:
        for idx, case in enumerate(cases, 1):
            case_started = time.perf_counter()
            relative = case["relative_dir"]
            expected_methods = (["refined_header", "refined_spatial_fusion"] if model is not None else ["refined_header"]) if a.mode == "refine" else (["initial_header", "initial_spatial_fusion"] if model is not None else ["initial_header"])
            already_done = all((relative, method) in done and done[(relative, method)].get("status") == "ok" for method in expected_methods)
            if already_done and a.mode == "refine":
                for name in (["header", "spatial_fusion"] if model is not None else ["header"]):
                    matrix_path = a.output / "transforms" / case["group"] / Path(relative).name / name / "rigid.txt"
                    if not matrix_path.is_file():
                        already_done = False
                        break
            if already_done:
                completed = done[(relative, expected_methods[0])]
                oracle_votes.append(bool(completed["oracle_preferred"]))
                objectives.append({"case": relative, "header": completed["initial_objective"],
                                   "oracle": completed["oracle_objective"],
                                   "oracle_preferred": bool(completed["oracle_preferred"])})
                print(f"[{idx}/{len(cases)}] Already complete: {relative}", flush=True)
                continue
            print(f"[{idx}/{len(cases)}] Loading {relative}", flush=True)
            slab_dir = a.data / relative
            sample = SavedSlabDataset(a.data, slab_dirs=[slab_dir])[0]
            slab = sample["slab"].to(device)
            slab_affine = sample["slab_affine"].to(device)
            t1 = sample["t1"].to(device)
            t1_affine = sample["t1_affine"].to(device)
            identity = torch.eye(4, device=device)
            mask_path = a.data / "data" / case["subject"] / "anat" / "T1_mask.nii.gz"
            brain_cpu, brain_affine = load_mask(str(mask_path))
            if not torch.allclose(brain_affine, sample["t1_affine"], atol=1e-3):
                raise ValueError(f"Brain mask / T1 affines differ for {relative}")
            brain = brain_cpu.to(device)
            points = voxel_centers(sample["slab_affine"].numpy(), sample["slab"].shape[-3:])
            oracle = fit_rigid(points[sample["mask"]], sample["truth"][sample["mask"]])
            oracle_t = torch.as_tensor(oracle, device=device, dtype=torch.float32)
            with torch.no_grad():
                t0 = time.perf_counter()
                cnn = model(slab, slab_affine, t1, t1_affine).matrix.detach() if model else None
                if device.type == "cuda":
                    torch.cuda.synchronize()
                cnn_time = time.perf_counter() - t0
            score_header = score_rigid_similarity(slab, slab_affine, t1, t1_affine, identity, config=config, t1_brain_mask=brain)
            score_oracle = score_rigid_similarity(slab, slab_affine, t1, t1_affine, oracle_t, config=config, t1_brain_mask=brain)
            preferred = bool(score_oracle < score_header)
            oracle_votes.append(preferred)
            objectives.append({"case": relative, "header": score_header, "oracle": score_oracle, "oracle_preferred": preferred})
            print(f"[{idx}/{len(cases)}] Objective header={score_header:.5f}, oracle={score_oracle:.5f}, preferred={preferred}", flush=True)
            starts = [("header", identity, 0.0)]
            if cnn is not None:
                starts.append(("spatial_fusion", cnn, cnn_time))
            for name, start, init_time in starts:
                method = ("refined_" if a.mode == "refine" else "initial_") + name
                key = (relative, method)
                if key in done and done[key].get("status") == "ok":
                    continue
                begin = time.perf_counter()
                try:
                    if a.mode == "refine":
                        candidate, diagnostics = refine_rigid(
                            slab, slab_affine, t1, t1_affine, config=config,
                            initial_matrix=start, t1_brain_mask=brain)
                    else:
                        candidate, diagnostics = start, {}
                    if device.type == "cuda":
                        torch.cuda.synchronize()
                    record = score_candidate(candidate, sample, method=method, elapsed=init_time + time.perf_counter() - begin,
                                             relative=relative, subject=case["subject"])
                    record.update({"oracle_preferred": preferred, "initial_objective": score_header,
                                   "oracle_objective": score_oracle, **diagnostics})
                    if a.mode == "refine":
                        matrix_dir = a.output / "transforms" / case["group"] / slab_dir.name / name
                        matrix_dir.mkdir(parents=True, exist_ok=True)
                        np.savetxt(matrix_dir / "rigid.txt", candidate.detach().cpu().numpy(), fmt="%.10g")
                except (ValueError, RuntimeError, FloatingPointError) as e:
                    record = {"case": relative, "subject": case["subject"], "method": method,
                              "status": "error", "error": str(e), "failure": True, "runtime_s": time.perf_counter() - begin}
                append_row(a.output / "cases.jsonl", record)
                done[key] = record
                run.log({"case_index": idx, f"{method}/mean_tre_mm": record.get("mean_tre_mm"),
                         f"{method}/failure": int(record["failure"]),
                         "score/oracle_preferred": int(preferred)})
                tre = record.get("mean_tre_mm")
                tre_text = f"{tre:.3f} mm" if tre is not None else "unavailable"
                print(f"[{idx}/{len(cases)}] {method}: {record['status']}, TRE={tre_text}, elapsed={time.perf_counter() - case_started:.1f}s", flush=True)
        print("All cases processed; writing summary", flush=True)
        selected = [done[(c["relative_dir"], m)] for c in cases for m in
                    (["refined_header", "refined_spatial_fusion"] if a.mode == "refine" else ["initial_header", "initial_spatial_fusion"])
                    if (c["relative_dir"], m) in done]
        report = print_summary(selected, a.output / "summary.json")
        report.update({"case_ids": [c["relative_dir"] for c in cases], "similarity": config.similarity,
                       "oracle_preferred_pct": 100.0 * sum(oracle_votes) / len(oracle_votes),
                       "mode": a.mode, "n_cases": len(cases)})
        (a.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
        (a.output / "objectives.json").write_text(json.dumps(objectives, indent=2) + "\n")
        for method_name, metrics in report.items():
            if not isinstance(metrics, dict):
                continue
            for field in ("median_tre_mm", "median_rotation_error_deg",
                          "median_centroid_translation_error_mm", "failure_pct",
                          "tre_below_2mm_pct", "median_runtime_s"):
                value = metrics.get(field)
                if value is not None:
                    run.summary[f"{method_name}/{field}"] = value
        run.log({"score/oracle_preferred_pct": report["oracle_preferred_pct"]})
    finally:
        run.finish()


if __name__ == "__main__":
    main()
