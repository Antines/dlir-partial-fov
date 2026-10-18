"""Per-subject preprocessing for the slab simulator (raw scanner space, no gradient-nonlinearity correction).

T1: N4 bias correction, 1 mm resampling, SynthStrip brain mask, FAST white matter (for BBR).
Field map: topup on a spin-echo pair with opposite phase encoding (PE) -> field (Hz) + mean corrected SE.
Each run: field aligned to the SBRef, distortion-corrected SBRef, and the whole-brain anchor
(corrected SBRef -> T1) with FSL BBR; QC = BBR restarted from random 2-4 deg / 2-4 mm perturbations of the anchor
must return to the same optimum (all within 0.5 mm).

Usage:
  python prep_subject.py hcp  <.../<subject>/unprocessed/3T> <out_dir>
  python prep_subject.py bids <T1w.nii.gz> <sbref.nii.gz> <se_1.nii.gz> <se_2.nii.gz> <out_dir> [<bridge_epi.nii.gz>]
  (BIDS sidecars; sbref = any 3D reference EPI volume; bridge = a volume of another whole-brain run, same PE)
"""
import json
import os
import subprocess
import sys
from pathlib import Path

import ants
import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.forward import unwarp
from slabsim.geometry import apply, random_rigid, voxel_centers
from slabsim.tools import ants_world, flirt_world, world_flirt

FSL = Path(os.environ['FSLDIR']) / 'share' / 'fsl' / 'bin'
ANTS = Path(sys.executable).parent                   # ANTs binaries of the dlir conda env
HCP_ECHO_SPACING = 0.00058                           # s, HCP 3T fMRI and spin-echo field maps (no in-plane acceleration)
HCP_RUNS = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']


def run(tool, *args):
    subprocess.run([str(tool)] + [str(a) for a in args], check=True, stdout=subprocess.DEVNULL)


def prep_t1(t1_raw, out):
    """1 mm N4-corrected T1, SynthStrip brain and mask, white-matter segmentation for BBR."""
    if (out / 'T1_wmseg.nii.gz').exists():
        return                                                              # already done (re-run of a subject)
    out.mkdir(parents=True, exist_ok=True)
    run(ANTS / 'N4BiasFieldCorrection', '-d', 3, '-i', t1_raw, '-o', out / 'T1_n4.nii.gz', '-s', 2)
    run(ANTS / 'ResampleImage', 3, out / 'T1_n4.nii.gz', out / 'T1.nii.gz', '1x1x1', 0, 4)
    run('docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-v', f'{out.resolve()}:/data',
        'freesurfer/synthstrip', '-i', '/data/T1.nii.gz', '-o', '/data/T1_brain.nii.gz', '-m', '/data/T1_mask.nii.gz')
    run(FSL / 'fast', '-t', 1, '-n', 3, '-o', out / 'T1_fast', out / 'T1_brain.nii.gz')
    run(FSL / 'fslmaths', out / 'T1_fast_pve_2.nii.gz', '-thr', 0.5, '-bin', out / 'T1_wmseg.nii.gz')


def field_map(se_files, pe_dirs, trts, out):
    """topup on spin-echo images with opposite PE (BIDS labels, e.g. 'j-' and 'j'): field (Hz) + mean corrected SE."""
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f, pe, trt in zip(se_files, pe_dirs, trts):
        vec = [0, 0, 0]
        vec['ijk'.index(pe[0])] = -1 if pe.endswith('-') else 1
        img = nib.load(f)
        rows += [f'{vec[0]} {vec[1]} {vec[2]} {trt:.6f}'] * (img.shape[3] if img.ndim == 4 else 1)
    (out / 'acqp.txt').write_text('\n'.join(rows) + '\n')
    run(FSL / 'fslmerge', '-t', out / 'se_all.nii.gz', *se_files)
    run(FSL / 'topup', f'--imain={out / "se_all.nii.gz"}', f'--datain={out / "acqp.txt"}', '--config=b02b0.cnf',
        f'--out={out / "topup"}', f'--fout={out / "field_hz.nii.gz"}', f'--iout={out / "se_corr.nii.gz"}')
    run(FSL / 'fslmaths', out / 'se_corr.nii.gz', '-Tmean', out / 'se_corr_mean.nii.gz')


def prep_run(sbref_file, pe_dir, trt, fmap_dir, t1_dir, out, pe_sign=None):
    """Field aligned to the SBRef, corrected SBRef and whole-brain anchor for one run; returns QC numbers.
    pe_sign: known PE sign in our convention; None = chosen from the data (step 1)."""
    out.mkdir(parents=True, exist_ok=True)
    sbref, field = nib.load(sbref_file), nib.load(fmap_dir / 'field_hz.nii.gz')
    raw, fdata, pe_axis = sbref.get_fdata(), field.get_fdata(), 'ijk'.index(pe_dir[0])

    # 1) PE sign in our convention: the one whose unwarped SBRef best matches the corrected spin-echo image. The match
    #    is made before the SBRef is aligned to the SE, so a large head move between them makes it a coin toss
    #    (HCP: 9/1200 runs, both correlations < 0.26): use a known sign where there is one; corr_by_sign stays as QC
    se = ants.resample_image_to_target(ants.image_read(str(fmap_dir / 'se_corr_mean.nii.gz')),
                                       ants.image_read(str(sbref_file))).numpy()
    mask = se > np.percentile(se, 70)
    corr = {s: float(np.corrcoef(unwarp(raw, sbref.affine, (fdata, field.affine), pe_axis, s, trt)[mask], se[mask])[0, 1])
            for s in (1, -1)}
    pe_sign = pe_sign or max(corr, key=corr.get)

    # 2) move the field to the SBRef head position: corrected SBRef -> corrected SE (both undistorted), 6 DOF
    first = unwarp(raw, sbref.affine, (fdata, field.affine), pe_axis, pe_sign, trt)
    nib.save(nib.Nifti1Image(first.astype(np.float32), sbref.affine), out / 'sbref_corr0.nii.gz')
    run(FSL / 'flirt', '-in', out / 'sbref_corr0.nii.gz', '-ref', fmap_dir / 'se_corr_mean.nii.gz', '-dof', 6,
        '-omat', out / 'sbref2se.mat')
    sbref2se = flirt_world(out / 'sbref2se.mat', out / 'sbref_corr0.nii.gz', fmap_dir / 'se_corr_mean.nii.gz')
    field_affine = np.linalg.inv(sbref2se) @ field.affine                   # field re-headered into the SBRef frame
    nib.save(nib.Nifti1Image(fdata.astype(np.float32), field_affine), out / 'field_hz_sbref.nii.gz')
    corrected = unwarp(raw, sbref.affine, (fdata, field_affine), pe_axis, pe_sign, trt)
    nib.save(nib.Nifti1Image(corrected.astype(np.float32), sbref.affine), out / 'sbref_corr.nii.gz')
    nib.save(nib.Nifti1Image(raw.astype(np.float32), sbref.affine), out / 'sbref_raw.nii.gz')

    # 3) whole-brain anchor: BBR (epi_reg), cross-checked with ANTs rigid mutual information
    run(FSL / 'epi_reg', f'--epi={out / "sbref_corr.nii.gz"}', f'--t1={t1_dir / "T1.nii.gz"}',
        f'--t1brain={t1_dir / "T1_brain.nii.gz"}', f'--wmseg={t1_dir / "T1_wmseg.nii.gz"}', f'--out={out / "bbr"}')
    bbr = flirt_world(out / 'bbr.mat', out / 'sbref_corr.nii.gz', t1_dir / 'T1_brain.nii.gz')
    run(ANTS / 'antsRegistration', '-d', 3, '-o', f'[{out / "ants_"}]', '-r',
        f'[{t1_dir / "T1_brain.nii.gz"},{out / "sbref_corr.nii.gz"},1]', '-t', 'Rigid[0.1]',
        '-m', f'MI[{t1_dir / "T1_brain.nii.gz"},{out / "sbref_corr.nii.gz"},1,32,Regular,0.25]',
        '-c', '[1000x500x250x0,1e-6,10]', '-s', '3x2x1x0vox', '-f', '8x4x2x1')
    mi = ants_world(out / 'ants_0GenericAffine.mat')
    np.savetxt(out / 'anchor_epi2t1.txt', bbr)

    # 4) anchor precision: BBR restarted from random 2-4 deg / 2-4 mm perturbations of the anchor (inside BBR's
    #    capture range) must return to the same optimum; distances to the header and MI starts are reported only
    brain = voxel_centers(sbref.affine, raw.shape)[mask]
    dist = lambda T: float(np.linalg.norm(apply(T, brain) - apply(bbr, brain), axis=1).mean())
    spread = anchor_stability(out / 'sbref_corr.nii.gz', t1_dir, bbr, brain.mean(0), out)
    qc = {'fmap': fmap_dir.name, 'pe_dir': pe_dir, 'pe_axis': pe_axis, 'pe_sign': pe_sign, 'trt': trt, 'corr_by_sign': corr,
          'pe_sign_data_agrees': pe_sign == max(corr, key=corr.get),
          'anchor_restart_mm': spread, 'anchor_ok': max(spread) < 0.5,
          'header_to_anchor_mm': dist(np.eye(4)), 'mi_to_anchor_mm': dist(mi),
          'field_hz_p99_abs': float(np.percentile(np.abs(fdata), 99))}
    (out / 'acq.json').write_text(json.dumps(qc, indent=1))
    return qc


def anchor_stability(epi, t1_dir, anchor, center, out, n=4, seed=0):
    """Mean distance (mm, brain voxels) of BBR solutions restarted from n random perturbations of the anchor."""
    rng = np.random.default_rng(seed)
    img = nib.load(epi)
    pts = voxel_centers(img.affine, img.shape)[img.get_fdata() > np.percentile(img.get_fdata(), 70)]
    spread = []
    for i in range(n):
        world_flirt(anchor @ random_rigid(rng, 2, 4, center), epi, t1_dir / 'T1.nii.gz', out / f'restart_{i}.mat')
        run(FSL / 'flirt', '-in', epi, '-ref', t1_dir / 'T1.nii.gz', '-dof', 6, '-cost', 'bbr',
            '-wmseg', t1_dir / 'T1_wmseg.nii.gz', '-init', out / f'restart_{i}.mat', '-omat', out / f'restart_{i}_bbr.mat',
            '-schedule', Path(os.environ['FSLDIR']) / 'etc' / 'flirtsch' / 'bbr.sch')
        T = flirt_world(out / f'restart_{i}_bbr.mat', epi, t1_dir / 'T1.nii.gz')
        spread.append(float(np.linalg.norm(apply(T, pts) - apply(anchor, pts), axis=1).mean()))
    return spread


def sidecar(nifti):
    return json.loads(Path(str(nifti).replace('.nii.gz', '.json')).read_text())


def prep_bridge(bridge_sbref, t1_dir, out):
    """Bridge for the bridged-BBR baseline: a raw whole-brain SBRef of another run (same session, shim and PE)
    registered to the T1 with BBR and no field map, as in common practice."""
    run(FSL / 'imcp', bridge_sbref, out / 'bridge_epi.nii.gz')
    run(FSL / 'epi_reg', f'--epi={out / "bridge_epi.nii.gz"}', f'--t1={t1_dir / "T1.nii.gz"}',
        f'--t1brain={t1_dir / "T1_brain.nii.gz"}', f'--wmseg={t1_dir / "T1_wmseg.nii.gz"}', f'--out={out / "bridge_bbr"}')
    np.savetxt(out / 'bridge_epi2t1.txt', flirt_world(out / 'bridge_bbr.mat', out / 'bridge_epi.nii.gz', t1_dir / 'T1.nii.gz'))


def hcp(raw_dir, out):
    """All rest runs of one HCP subject; one topup per unique spin-echo pair; bridges chosen from the scan CSV."""
    from hashlib import md5
    from download_hcp import bridges
    subject = raw_dir.parts[-3]                                             # .../<subject>/unprocessed/3T
    prep_t1(raw_dir / 'T1w_MPR1' / f'{subject}_3T_T1w_MPR1.nii.gz', out / 'anat')
    bridge_runs = bridges(raw_dir / f'{subject}_3T.csv')
    for name in HCP_RUNS:
        run_dir = raw_dir / name
        se = [run_dir / f'{subject}_3T_SpinEchoFieldMap_{d}.nii.gz' for d in ('LR', 'RL')]
        trt = HCP_ECHO_SPACING * (nib.load(se[0]).shape[0] - 1)           # PE along the first voxel axis (x)
        fmap = out / 'fmaps' / md5(se[0].read_bytes() + se[1].read_bytes()).hexdigest()[:10]
        if not (fmap / 'field_hz.nii.gz').exists():
            field_map(se, ['i-', 'i'], [trt, trt], fmap)                    # HCP pipelines: LR = x-, RL = x
        sign = -1 if name.endswith('LR') else 1           # HCP LR = x-, RL = x: our sign in 1191/1200 runs (step 1)
        acq = out / name / 'acq.json'
        if acq.exists() and json.loads(acq.read_text())['pe_sign'] == sign:
            continue                                                        # done (re-run fixes only a wrong sign)
        qc = prep_run(run_dir / f'{subject}_3T_{name}_SBRef.nii.gz', 'i-' if name.endswith('LR') else 'i', trt,
                      fmap, out / 'anat', out / name, pe_sign=sign)
        if name in bridge_runs:
            task = bridge_runs[name]
            prep_bridge(raw_dir / task / f'{subject}_3T_{task}_SBRef.nii.gz', out / 'anat', out / name)
        print(subject, name, qc, flush=True)


if __name__ == '__main__':
    if sys.argv[1] == 'hcp':
        hcp(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        t1, sbref, se1, se2, out = map(Path, sys.argv[2:7])
        prep_t1(t1, out / 'anat')
        meta = [sidecar(f) for f in (se1, se2)]
        field_map([se1, se2], [m['PhaseEncodingDirection'] for m in meta], [m['TotalReadoutTime'] for m in meta],
                  out / 'run' / 'fmap')
        m = sidecar(sbref)
        print(prep_run(sbref, m['PhaseEncodingDirection'], m['TotalReadoutTime'], out / 'run' / 'fmap',
                       out / 'anat', out / 'run'))
        if len(sys.argv) > 7:
            prep_bridge(Path(sys.argv[7]), out / 'anat', out / 'run')
