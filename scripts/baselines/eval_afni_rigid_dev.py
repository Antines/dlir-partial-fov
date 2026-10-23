#!/usr/bin/env python3
"""Evaluate AFNI lpc+ZZ 6-DOF on a locked, development-only partial-FOV subset."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts"))

from dl.rigid_dev_common import append_row, assert_rigid, ensure_manifest, load_rows, print_summary
from dl.rigid_dev_wandb import start_tracking
from slabsim.evaluation import displacement_error, linear_estimate, rigid_errors
from slabsim.geometry import voxel_centers
from slabsim.samples import load_sample
from slabsim.tools import plumb


def afni_command(backend: str, anat: Path, case: Path, threads: int) -> list[str]:
    arguments = [
        "align_epi_anat.py", "-anat", "/anat/T1_brain_plumb.nii.gz", "-anat_has_skull", "no",
        "-epi", "/out/slab_plumb.nii.gz", "-epi_base", "0", "-epi_strip", "None",
        "-cost", "lpc+ZZ", "-partial_coverage", "-large_move", "-rigid_body",
        "-volreg", "off", "-tshift", "off", "-deoblique", "off",
        "-suffix", "_al", "-output_dir", "/out", "-overwrite",
    ]
    if backend == "docker":
        if not shutil.which("docker"):
            raise EnvironmentError("Docker is unavailable. Use --backend native on a system with AFNI installed.")
        return [
            "docker", "run", "--rm", "-u", f"{os.getuid()}:{os.getgid()}",
            "-e", f"OMP_NUM_THREADS={threads}",
            "-v", f"{anat.resolve()}:/anat:ro", "-v", f"{case.resolve()}:/out",
            "-w", "/out", "afni/afni_make_build", *arguments,
        ]
    if backend != "native":
        raise ValueError(f"Unrecognized backend: {backend}")
    if not shutil.which("align_epi_anat.py"):
        raise EnvironmentError("AFNI executable align_epi_anat.py not found in PATH")
    return [str(a).replace("/anat/T1_brain_plumb.nii.gz", str(anat / "T1_brain_plumb.nii.gz"))
                       .replace("/out/slab_plumb.nii.gz", str(case / "slab_plumb.nii.gz"))
                       .replace("/out", str(case)) for a in arguments]


def afni_matrix(path: Path) -> np.ndarray:
    values = np.asarray(np.loadtxt(path), dtype=float)
    if values.size != 12:
        raise ValueError(f"Expected AFNI 3x4 matrix at {path}, got {values.shape}")
    m = np.eye(4)
    m[:3, :] = values.reshape(3, 4)
    flip = np.diag([-1, -1, 1, 1])
    return assert_rigid(flip @ m @ flip)


def run_afni(case_dir: Path, anat_source: Path, cache_root: Path, output: Path, backend: str, threads: int) -> np.ndarray:
    output.mkdir(parents=True, exist_ok=True)
    cache_root.mkdir(parents=True, exist_ok=True)
    anatomical = cache_root / "T1_brain_plumb.nii.gz"
    if not anatomical.is_file():
        plumb(anat_source, anatomical, voxel=1.0)
    slab = output / "slab_plumb.nii.gz"
    if not slab.is_file():
        plumb(case_dir / "slab.nii.gz", slab)
    cmd = afni_command(backend, cache_root, output, threads)
    (output / "command.json").write_text(json.dumps(cmd, indent=2) + "\n")
    with (output / "afni.log").open("w") as log:
        subprocess.run(cmd, check=True, stdout=log, stderr=subprocess.STDOUT)
    return afni_matrix(output / "T1_brain_plumb_al_mat.aff12.1D")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--max-cases", type=int, default=36)
    parser.add_argument("--backend", choices=["native", "docker"], default="native")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--wandb-project", default="neuroml-dlir")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = ensure_manifest(args.data, args.manifest, count=args.max_cases)
    if args.backend == "native" and not shutil.which("align_epi_anat.py"):
        raise EnvironmentError("AFNI unavailable; install AFNI before running this benchmark")
    if args.backend == "docker" and not shutil.which("docker"):
        raise EnvironmentError("Docker unavailable; run on an AFNI-enabled workstation instead")
    print(f"AFNI rigid benchmark: {len(cases)} locked cases; backend={args.backend}", flush=True)
    run = start_tracking(args.output, project=args.wandb_project, name="rigid-afni-6dof-dev-v1", config={"backend": args.backend, "cases": len(cases)})
    output_file = args.output / "cases.jsonl"
    existing = load_rows(output_file)
    try:
        for i, item in enumerate(cases, 1):
            relative = item["relative_dir"]
            key = (relative, "afni_rigid_6dof")
            if key in existing and existing[key].get("status") == "ok":
                continue
            print(f"[{i}/{len(cases)}] AFNI starting {relative}", flush=True)
            sample_dir = args.data / relative
            anat_dir = args.data / "data" / item["subject"] / "anat"
            case_output = args.output / "registration" / item["group"] / sample_dir.name
            cache = args.output / "anat_cache" / item["subject"]
            t0 = time.perf_counter()
            record = {"case": relative, "subject": item["subject"], "method": "afni_rigid_6dof", "status": "error"}
            try:
                matrix = run_afni(sample_dir, anat_dir / "T1_brain.nii.gz", cache, case_output, args.backend, args.threads)
                image, affine, truth, mask = load_sample(sample_dir)
                estimated = linear_estimate(matrix, affine, image.shape)
                error = displacement_error(estimated, truth, mask)
                pose = rigid_errors(estimated, truth, voxel_centers(affine, image.shape), mask)
                record.update({"status": "ok", "mean_tre_mm": error["mean"],
                               "rot_deg": pose["rot_deg"], "trans_mm": pose["trans_mm"],
                               "failure": bool(pose["rot_deg"] > 2 or pose["trans_mm"] > 2)})
                np.savetxt(case_output / "rigid.txt", matrix, fmt="%.10g")
            except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
                record.update({"failure": True, "error": str(exc)})
            record["runtime_s"] = time.perf_counter() - t0
            append_row(output_file, record)
            existing[key] = record
            run.log({"case_index": i, "afni/mean_tre_mm": record.get("mean_tre_mm"),
                     "afni/failure": int(record["failure"]), "afni/runtime_s": record["runtime_s"]})
            print(f"[{i}/{len(cases)}] AFNI {record['status']}; elapsed={record['runtime_s']:.1f}s; {relative}", flush=True)
        print_summary([existing[(c["relative_dir"], "afni_rigid_6dof")] for c in cases if (c["relative_dir"], "afni_rigid_6dof") in existing], args.output / "summary.json")
    finally:
        run.finish()


if __name__ == "__main__":
    main()
