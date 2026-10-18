"""Visual anchor QC: the anchored, distortion-corrected SBRef in T1 space (bbr.nii.gz) with the
T1 white-matter boundary in red, three orthogonal mid-slices per run, on contact sheets of 24 runs.
Usage: python qc_overlays.py <prep_root> <out_dir>
"""
import os
import subprocess
import sys
from multiprocessing import Pool
from pathlib import Path

from PIL import Image, ImageDraw

FSL = Path(os.environ['FSLDIR']) / 'share' / 'fsl' / 'bin'
H = 200                                                      # height of each view on the sheet


def strip(run):
    out = OUT / f'{run.parent.name}_{run.name}.png'
    if not out.exists():
        views = [OUT / f'.{out.stem}_{a}.png' for a in 'xyz']
        subprocess.run([FSL / 'slicer', run / 'bbr.nii.gz', run.parent / 'anat' / 'T1_wmseg.nii.gz',
                        '-x', '0.5', views[0], '-y', '0.5', views[1], '-z', '0.5', views[2]], check=True)
        ims = [Image.open(v).convert('RGB') for v in views]
        ims = [im.resize((round(im.width * H / im.height), H)) for im in ims]
        row = Image.new('RGB', (sum(im.width for im in ims), H + 16))
        for i, im in enumerate(ims):
            row.paste(im, (sum(m.width for m in ims[:i]), 16))
        ImageDraw.Draw(row).text((4, 2), out.stem, fill='yellow')
        row.save(out)
        for v in views:
            v.unlink()
    return out


if __name__ == '__main__':
    prep, OUT = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    runs = sorted(p.parent for p in prep.glob('*/rfMRI_*/bbr.nii.gz'))
    with Pool(4) as pool:
        strips = pool.map(strip, runs)
    for page in range(0, len(strips), 24):                   # 3 columns x 8 rows
        ims = [Image.open(s) for s in strips[page:page + 24]]
        w, h = max(im.width for im in ims), ims[0].height
        sheet = Image.new('RGB', (3 * w, 8 * h))
        for i, im in enumerate(ims):
            sheet.paste(im, ((i % 3) * w, (i // 3) * h))
        sheet.save(OUT / f'sheet_{page // 24 + 1:02d}.png')
    print(len(strips), 'runs,', (len(strips) + 23) // 24, 'sheets in', OUT)
