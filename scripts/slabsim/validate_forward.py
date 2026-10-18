"""Realism check of the forward model on real data: re-distort the topup-corrected spin-echo image with the measured
field and compare it with the real distorted spin echoes of both phase-encoding polarities.

Per field map (a fmaps/<id> folder of prep_subject.py: se_all, acqp.txt, se_corr_mean, field_hz) and polarity: the
correlation inside the head between the real image and (a) the undistorted image (how much distortion there is),
(b) our simulation (how much of it the model explains). Head motion between the volumes is taken from topup's
estimate (topup_movpar.txt). Not independent of topup, which fitted the same model to these images; it checks our
implementation, sign convention and inversion, and flags field maps that do not reproduce their own spin echoes.
The lower correlation of the two polarities is written as 'field_check_corr' into the acq.json of every run that uses
the field map (inclusion rule: slabsim.dataset.passes_qc).
Usage: python validate_forward.py <fmap_dir> [<fmap_dir> ...] > forward_check.csv
"""
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.forward import simulate_slab
from slabsim.geometry import apply, rigid

if __name__ == '__main__':
    print('fmap,pe_axis,pe_sign,motion_mm,motion_deg,corr_undistorted,corr_simulated', flush=True)
    for fmap in map(Path, sys.argv[1:]):
        acqp = np.loadtxt(fmap / 'acqp.txt', ndmin=2)            # per volume: PE vector (topup = our sign) + readout time
        movpar = np.loadtxt(fmap / 'topup_movpar.txt', ndmin=2)  # per volume: translation (mm), rotation (rad) vs volume 0
        se = nib.load(fmap / 'se_all.nii.gz')
        obj, field = nib.load(fmap / 'se_corr_mean.nii.gz'), nib.load(fmap / 'field_hz.nii.gz')
        obj_data = obj.get_fdata()
        head = obj_data > np.percentile(obj_data, 60)
        sim_corr = []
        for sign in (-1, 1):
            volume = int(np.flatnonzero(acqp[:, :3].sum(1) == sign)[0])        # first volume of this polarity
            pe_axis, trt = int(np.argmax(np.abs(acqp[volume, :3]))), float(acqp[volume, 3])
            real = se.dataobj[..., volume]
            # head pose of this volume from topup's estimate: FSL mm axes -> scanner axes, rotation about the centre
            # (convention checked on HCP volumes with 1-7 mm motion, where it raises the correlation)
            F = np.diag([-1.0, 1, 1]) if np.linalg.det(se.affine[:3, :3]) < 0 else np.eye(3)
            t, rot = F @ movpar[volume, :3], -np.linalg.det(F) * F @ np.degrees(movpar[volume, 3:])
            pose = np.linalg.inv(rigid(rot, t, apply(se.affine, (np.array(real.shape) - 1) / 2)))
            sim, _, _, valid = simulate_slab((obj_data, obj.affine), (field.get_fdata(), field.affine), se.affine,
                                             real.shape, head_pose=pose, pe_axis=pe_axis, pe_sign=sign, trt=trt)
            m = head & valid
            r = lambda a: np.corrcoef(a[m], real[m])[0, 1]
            sim_corr.append(r(sim))
            print(f'{fmap.parent.parent.name}/{fmap.name},{pe_axis},{sign},{np.linalg.norm(t):.2f},'
                  f'{np.linalg.norm(rot):.2f},{r(obj_data):.4f},{r(sim):.4f}', flush=True)
        for acq in fmap.parent.parent.glob('rfMRI_*/acq.json'):          # the runs that use this field map
            q = json.loads(acq.read_text())
            if q.get('fmap') == fmap.name:
                q['field_check_corr'] = round(min(sim_corr), 4)
                acq.with_suffix('.tmp').write_text(json.dumps(q, indent=1))
                acq.with_suffix('.tmp').replace(acq)                        # atomic: other jobs may read it
