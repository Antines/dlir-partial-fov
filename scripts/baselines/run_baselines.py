"""Run classical baselines on saved slab samples and score them against the ground truth.

Every method gets the same inputs: the slab (moving), the subject's 1 mm T1 (fixed, head and brain) and its
white-matter segmentation, and starts from the scanner header unless its standard pipeline does otherwise.
Usage: python run_baselines.py <anat_dir> <sample_dir> [<sample_dir> ...]   -> results.json in each sample dir
"""
import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from slabsim.evaluation import coordinate_maps, displacement_error, fit_rigid, linear_estimate, rigid_errors
from slabsim.geometry import voxel_centers
from slabsim.samples import load_sample
from slabsim.tools import ants_world, flirt_world, plumb, world_flirt

FSL = Path(os.environ['FSLDIR']) / 'share' / 'fsl' / 'bin'
ANTS = Path(sys.executable).parent
THREADS = os.environ.get('OMP_NUM_THREADS', '4')         # also passed into the AFNI / SynthMorph containers
GPU = os.environ.get('DLIR_GPU') == '1'                   # SynthMorph on an NVIDIA GPU (docker --gpus all)
CLEAN = os.environ.get('DLIR_CLEAN') == '1'               # delete a method's work folder once it is scored


def run(tool, *args):
    subprocess.run([str(tool)] + [str(a) for a in args], check=True, stdout=subprocess.DEVNULL)


def header(slab, anat, out):
    """No registration: the slab stays where its scanner header puts it."""
    return np.eye(4)


def rigid_oracle(slab, anat, out):
    """Reference, not a baseline: the least-squares rigid fit to the ground truth. Its TRE is the floor for any
    rigid method; what remains is the distortion."""
    image, affine, truth, mask = load_sample(slab.parent)
    return fit_rigid(voxel_centers(affine, image.shape)[mask], truth[mask])


def affine_oracle(slab, anat, out):
    """Reference, not a baseline: the least-squares affine (12 DOF) fit to the ground truth, the floor for affine
    methods such as AFNI's default. It can beat the rigid oracle because scaling and shear absorb some distortion."""
    image, affine, truth, mask = load_sample(slab.parent)
    p = voxel_centers(affine, image.shape)[mask]
    M = np.eye(4)
    M[:3] = np.linalg.lstsq(np.c_[p, np.ones(len(p))], truth[mask], rcond=None)[0].T
    return M


def fsl_bbr(slab, anat, out):
    """FSL boundary-based registration (6 DOF) started from the scanner header."""
    world_flirt(np.eye(4), slab, anat / 'T1.nii.gz', out / 'header.mat')
    run(FSL / 'flirt', '-in', slab, '-ref', anat / 'T1.nii.gz', '-dof', 6, '-cost', 'bbr',
        '-wmseg', anat / 'T1_wmseg.nii.gz', '-init', out / 'header.mat', '-omat', out / 'bbr.mat',
        '-schedule', Path(os.environ['FSLDIR']) / 'etc' / 'flirtsch' / 'bbr.sch')
    return flirt_world(out / 'bbr.mat', slab, anat / 'T1.nii.gz')


def epi_reg(slab, anat, out):
    """FSL epi_reg as distributed: its own FLIRT initialisation, then BBR."""
    run(FSL / 'epi_reg', f'--epi={slab}', f'--t1={anat / "T1.nii.gz"}', f'--t1brain={anat / "T1_brain.nii.gz"}',
        f'--wmseg={anat / "T1_wmseg.nii.gz"}', f'--out={out / "epireg"}')
    return flirt_world(out / 'epireg.mat', slab, anat / 'T1.nii.gz')


def ants_rigid(slab, anat, out):
    """ANTs rigid registration with mutual information, started from the header (identity)."""
    run(ANTS / 'antsRegistration', '-d', 3, '-o', f'[{out / "ants_"}]', '-t', 'Rigid[0.1]',
        '-m', f'MI[{anat / "T1_brain.nii.gz"},{slab},1,32,Regular,0.25]',
        '-c', '[1000x500x250x0,1e-6,10]', '-s', '3x2x1x0vox', '-f', '8x4x2x1')
    return ants_world(out / 'ants_0GenericAffine.mat')


def bridged_bbr(slab, anat, out):
    """Standard practice when a whole-brain EPI with the same parameters exists (FSL FLIRT guide):
    slab -> whole-brain EPI (6 DOF, same contrast and distortion) -> T1 with that EPI's own BBR.

    The bridge (bridge_epi.nii.gz + bridge_epi2t1.txt, from prep_subject.py) lives in the slab's source run folder;
    it is a DIFFERENT run than the slab (a same-session task SBRef with the same PE), so its BBR is not the anchor.
    """
    run_dir = anat.parent / Path(json.loads((slab.parent / 'params.json').read_text())['run']).name   # data may move
    bridge = run_dir / 'bridge_epi.nii.gz'
    if not bridge.exists():
        raise FileNotFoundError(bridge)
    world_flirt(np.eye(4), slab, bridge, out / 'header.mat')
    run(FSL / 'flirt', '-in', slab, '-ref', bridge, '-dof', 6, '-cost', 'normcorr', '-init', out / 'header.mat',
        '-nosearch', '-omat', out / 'slab2bridge.mat')
    return np.loadtxt(run_dir / 'bridge_epi2t1.txt') @ flirt_world(out / 'slab2bridge.mat', slab, bridge)


def afni_lpc(slab, anat, out):
    """AFNI align_epi_anat.py, lpc+ZZ cost, partial-coverage settings, on axis-aligned copies of both images.
    Its default transform is affine (12 DOF, '-warp aff'), kept as distributed."""
    plumb(slab, out / 'slab_plumb.nii.gz')
    if not (anat / 'T1_brain_plumb.nii.gz').exists():
        plumb(anat / 'T1_brain.nii.gz', anat / 'T1_brain_plumb.nii.gz', voxel=1.0)
    run('docker', 'run', '--rm', '-u', f'{os.getuid()}:{os.getgid()}', '-e', f'OMP_NUM_THREADS={THREADS}',
        '-v', f'{anat.resolve()}:/anat:ro', '-v', f'{out.resolve()}:/out', '-w', '/out', 'afni/afni_make_build',
        'align_epi_anat.py',
        '-anat', '/anat/T1_brain_plumb.nii.gz', '-anat_has_skull', 'no', '-epi', '/out/slab_plumb.nii.gz',
        '-epi_base', 0, '-epi_strip', 'None', '-cost', 'lpc+ZZ', '-partial_coverage', '-large_move',
        '-volreg', 'off', '-tshift', 'off', '-deoblique', 'off', '-suffix', '_al', '-output_dir', '/out', '-overwrite')
    M = np.eye(4)
    M[:3] = np.loadtxt(out / 'T1_brain_plumb_al_mat.aff12.1D').reshape(3, 4)   # EPI (base) -> anat (source), RAI
    flip = np.diag([-1, -1, 1, 1])                                            # AFNI RAI <-> RAS
    return flip @ M @ flip


def coordinate_files(anat):
    """x, y, z world-coordinate images on the 1 mm T1 grid (written once per subject)."""
    files = [anat / f'T1_coord_{a}.nii.gz' for a in 'xyz']
    if not files[0].exists():
        t1 = nib.load(anat / 'T1.nii.gz')
        coords = coordinate_maps(t1.affine, t1.shape)
        for i, f in enumerate(files):
            nib.save(nib.Nifti1Image(coords[..., i], t1.affine), f)
    return files


def synthmorph(model):
    """SynthMorph (zero-shot, CPU or GPU) registering the slab to the T1; its T1 -> slab transform is applied to the
    T1 coordinate images, which gives the estimated T1 position of every slab voxel (also for deformable warps)."""
    def method(slab, anat, out):
        coords = coordinate_files(anat)
        trans = 't1_to_slab.lta' if model in ('rigid', 'affine') else 't1_to_slab.nii.gz'
        docker = ['docker', 'run', '--rm', *(['--gpus', 'all'] if GPU else []), '-u', f'{os.getuid()}:{os.getgid()}',
                  '-v', f'{anat.resolve()}:/anat:ro',
                  '-v', f'{slab.parent.resolve()}:/sample:ro', '-v', f'{out.resolve()}:/out', 'freesurfer/synthmorph']
        device = ['-g'] if GPU else ['-j', THREADS]
        extent = [] if model in ('rigid', 'affine') else ['-e', 192]     # joint at 256^3: OOM on CPU and on a 24 GB GPU
        pairs = [x for a, f in zip('xyz', coords) for x in (f'/anat/{f.name}', f'/out/coord_{a}.nii.gz')]
        with open('/tmp/dlir_synthmorph.lock', 'w') as lock:             # one container at a time: TensorFlow
            fcntl.flock(lock, fcntl.LOCK_EX)                              # reserves the whole GPU
            run(*docker, 'register', '-m', model, *device, *extent, '-T', f'/out/{trans}', f'/sample/{slab.name}',
                '/anat/T1.nii.gz')
            run(*docker, 'apply', f'/out/{trans}', *pairs)
        return np.stack([nib.load(out / f'coord_{a}.nii.gz').get_fdata() for a in 'xyz'], axis=-1)
    return method


METHODS = {'header': header, 'rigid_oracle': rigid_oracle, 'affine_oracle': affine_oracle, 'fsl_bbr': fsl_bbr, 'epi_reg': epi_reg, 'ants_rigid_mi': ants_rigid,
           'bridged_bbr': bridged_bbr, 'afni_lpc': afni_lpc,
           'synthmorph_rigid': synthmorph('rigid'), 'synthmorph_affine': synthmorph('affine'),
           'synthmorph_joint': synthmorph('joint')}

if __name__ == '__main__':
    anat = Path(sys.argv[1])
    if os.environ.get('DLIR_METHODS'):                        # e.g. DLIR_METHODS=header,fsl_bbr,afni_lpc
        METHODS = {k: METHODS[k] for k in os.environ['DLIR_METHODS'].split(',')}
    for sample_dir in map(Path, sys.argv[2:]):
        done = sample_dir / 'results.json'
        results = json.loads(done.read_text()) if done.exists() else {}
        todo = {k: v for k, v in METHODS.items() if 'failed' in results.get(k, {'failed': ''})}   # resume: keep results, retry failures
        if not todo:
            continue
        image, affine, truth, mask = load_sample(sample_dir)
        points = voxel_centers(affine, image.shape)
        for name, method in todo.items():
            out = sample_dir / name
            out.mkdir(exist_ok=True)
            t0 = time.time()
            try:
                world = method(sample_dir / 'slab.nii.gz', anat, out)
            except (subprocess.CalledProcessError, OSError) as e:      # tool failure or missing bridge EPI
                results[name] = {'failed': str(e)}
                continue
            estimate = world if world.shape != (4, 4) else linear_estimate(world, affine, image.shape)
            results[name] = {'runtime_s': time.time() - t0, **displacement_error(estimate, truth, mask),
                             **rigid_errors(estimate, truth, points, mask)}
            if CLEAN:
                shutil.rmtree(out, ignore_errors=True)
        done.write_text(json.dumps(results, indent=1))
        print(sample_dir.name, {k: round(v.get('mean', np.nan), 2) for k, v in results.items()})
