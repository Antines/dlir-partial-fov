#!/usr/bin/env python3
"""Export validated rigid matrices and acquisition metadata for the deformable stage."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np


def rigid_matrix(a: np.ndarray, tolerance: float = 1e-3) -> np.ndarray:
    a = np.asarray(a, np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all():
        raise ValueError('Rigid matrix must be a finite 4x4 array')
    if not np.allclose(a[-1], [0, 0, 0, 1], atol=1e-8):
        raise ValueError('Invalid homogeneous row')
    r = a[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=tolerance) or abs(np.linalg.det(r) - 1) > tolerance:
        raise ValueError('Matrix contains shear, scale, or reflection')
    return a


def corrected_geometry(rigid: np.ndarray, slab_affine: np.ndarray, pe_axis: int, pe_sign: int, trt: float) -> dict:
    r = rigid_matrix(rigid)
    a = np.asarray(slab_affine, np.float64)
    if a.shape != (4, 4) or not np.isfinite(a).all() or not np.allclose(a[3], [0, 0, 0, 1]):
        raise ValueError('Invalid slab affine')
    if not isinstance(pe_axis, (int, np.integer)) or pe_axis not in (0, 1, 2):
        raise ValueError('PE axis must be 0, 1, or 2 in the slab voxel grid')
    if pe_sign not in (-1, 1):
        raise ValueError('PE sign must be -1 or +1')
    if not np.isfinite(trt) or trt <= 0:
        raise ValueError('Readout time must be finite and positive')
    corrected = r @ a
    pe_step = corrected[:3, pe_axis]
    n = np.linalg.norm(pe_step)
    if n < 1e-8:
        raise ValueError('Degenerate PE axis in corrected affine')
    u = pe_step / n
    return {
        'rigid_matrix': r.tolist(),
        'original_slab_affine': a.tolist(),
        'corrected_affine': corrected.tolist(),
        'pe_axis': int(pe_axis),
        'pe_sign': int(pe_sign),
        'trt': float(trt),
        'pe_world_unit_vector': u.tolist(),
        'signed_pe_world_unit_vector': (pe_sign * u).tolist(),
    }


def atomic_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(tmp, path)


def sha256_file(path: Path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build_handoff(data: Path, manifest: dict, result_rows: dict, transform_root: Path,
                  output: Path, limit: int | None = None) -> dict:
    if manifest.get('data_split') != 'validation':
        raise ValueError('Export only the development validation split')
    cases = manifest['cases']
    if len(cases) != len({c['relative_dir'] for c in cases}):
        raise ValueError('Manifest contains duplicate cases')
    if limit is not None:
        cases = cases[:limit]
    if not cases:
        raise ValueError('No cases to export')
    import nibabel as nib
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for i, case in enumerate(cases, start=1):
        relative = Path(case['relative_dir'])
        if relative.is_absolute() or '..' in relative.parts or len(relative.parts) != 3 or relative.parts[0] != 'test':
            raise ValueError(f'Invalid case directory: {relative}')
        case_id = relative.as_posix()
        if case_id not in result_rows or result_rows[case_id].get('status') != 'ok':
            raise ValueError(f'Missing successful metric record: {case_id}')
        params_file = data / relative / 'params.json'
        image_file = data / relative / 'slab.nii.gz'
        transform = transform_root / relative.parts[1] / relative.parts[2] / 'header' / 'rigid.txt'
        for path in (params_file, image_file, transform):
            if not path.is_file():
                raise FileNotFoundError(path)
        params = json.loads(params_file.read_text())
        image = nib.load(str(image_file))
        matrix = rigid_matrix(np.loadtxt(transform))
        geometry = corrected_geometry(matrix, image.affine, params['pe_axis'], params['pe_sign'], params['trt'])
        dest = output / 'cases' / relative.parts[1] / relative.parts[2]
        dest.mkdir(parents=True, exist_ok=True)
        metadata = {
            'case': case_id, 'subject': case['subject'], 'source_slab': str(image_file),
            'source_rigid_transform': str(transform), 'source_rigid_sha256': sha256_file(transform),
            'voxel_shape_xyz': list(image.shape[:3]),
            'transform_convention': 'RAS-mm world residual; corrected_affine = rigid_matrix @ slab_affine',
            'metadata_convention': 'pe_axis is an index in original slab voxel coordinates',
            **geometry,
        }
        dest_meta = dest / 'metadata.json'
        if dest_meta.is_file():
            if json.loads(dest_meta.read_text()) != metadata:
                raise ValueError(f'Saved handoff mismatches current source: {dest_meta}')
            if i == 1 or i % 25 == 0 or i == len(cases):
                print(f'[{i}/{len(cases)}] Existing: {case_id}', flush=True)
        else:
            np.savetxt(dest / 'rigid.txt', matrix, fmt='%.12g')
            atomic_json(dest_meta, metadata)
            if i == 1 or i % 25 == 0 or i == len(cases):
                print(f'[{i}/{len(cases)}] Exported: {case_id}', flush=True)
        records.append(metadata)
    header = {
        'split': 'validation', 'n_cases': len(records),
        'subjects': len({r['subject'] for r in records}),
        'data_usage': 'Rigid outputs only; no truth or evaluation masks exported',
        'transform': 'slab-to-T1 RAS-mm residual, applied on the left of slab NIfTI affine',
    }
    atomic_json(output / 'handoff_summary.json', header)
    target = output / 'handoff.jsonl'
    tmp = target.with_name(target.name + '.tmp')
    with tmp.open('w') as f:
        for record in records:
            f.write(json.dumps(record, allow_nan=False) + '\n')
    os.replace(tmp, target)
    return header


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--results', type=Path, required=True)
    p.add_argument('--transforms', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--limit', type=int)
    a = p.parse_args()
    rows = {}
    for line in a.results.read_text().splitlines():
        r = json.loads(line)
        if r.get('method') == 'refined_header':
            rows[r['case']] = r
    report = build_handoff(a.data, json.loads(a.manifest.read_text()), rows, a.transforms, a.output, a.limit)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
