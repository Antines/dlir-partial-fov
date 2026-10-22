"""Figures and the W&B Report of the deformable stage, from the outputs of run_all.sh and analysis.py.

Run from scripts/ after all experiments and analyses:  python -m deformable.report [--url REPORT_URL]
Saves the figures to runs/deformable/report/, uploads them to the W&B run `report_figures` and creates the report
(with --url: replaces the content of that report). Needs `pip install wandb-workspaces`.
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import wandb
import wandb_workspaces.reports.v2 as wr

REPO = Path(__file__).resolve().parents[2]
ENTITY, PROJECT = 'chetverkin-ai', 'dlir-deformable'
OUT = REPO / 'runs' / 'deformable'
FIG = OUT / 'report'
CONFIGS = sorted(p.stem for p in (REPO / 'scripts' / 'deformable' / 'configs').glob('*.yaml'))

BLUE, ORANGE, AQUA, GREY = '#2a78d6', '#eb6834', '#1baf7a', '#b4b3ad'
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({'font.size': 10, 'axes.edgecolor': GRID, 'axes.labelcolor': MUTED, 'xtick.color': MUTED,
                     'ytick.color': INK, 'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.titlelocation': 'left',
                     'figure.dpi': 150, 'savefig.bbox': 'tight'})


def summary(name):
    return json.loads((OUT / name / 'val_summary.json').read_text())


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def references(s):
    """Reference medians on the 1,080 validation slabs, from the summary of a run."""
    return [('no deformation', s['no_deformation']), ('AFNI lpc+ZZ', s['base_afni_lpc']),
            ('rigid oracle', s['base_rigid_oracle']), ('affine oracle', s['base_affine_oracle'])]


def style(ax, xlabel):
    ax.set_xlabel(xlabel)
    ax.grid(axis='x', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ('top', 'right', 'left'):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis='y', length=0)


def bars(ax, labels, values, final, xlabel, refs=(), fmt='{:.2f}'.format):
    """Horizontal bars, top to bottom, value at the end; the final model in blue, the rest grey. Reference values:
    dashed vertical lines behind the bars, labelled above them (in up to three rows, so labels never overlap)."""
    n = len(values)
    y = np.arange(n)[::-1]
    xmax = 1.15 * max([*values, *[v for _, v in refs], 1e-3])
    ax.barh(y, values, height=0.62, color=[BLUE if i == final else GREY for i in range(n)], zorder=2)
    for yi, v in zip(y, values):
        inside = v > 0.3 * xmax
        ax.text(v - 0.012 * xmax if inside else v + 0.012 * xmax, yi, fmt(v), va='center', fontsize=9,
                ha='right' if inside else 'left', color='white' if inside else INK, zorder=3)
    ax.set_xlim(0, xmax)
    renderer, placed, rows = ax.figure.canvas.get_renderer(), [], 0
    for name, v in sorted(refs, key=lambda r: r[1]):
        ax.axvline(v, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)), zorder=1)
        for row in range(3):
            t = ax.text(v, n - 0.55 + 0.5 * row, f'{name} {v:.2f}', ha='center', va='bottom', fontsize=8, color=MUTED,
                        bbox=dict(facecolor='white', edgecolor='none', pad=1))
            box = t.get_window_extent(renderer).expanded(1.08, 1.0)
            if not any(box.overlaps(b) for b in placed) or row == 2:
                placed.append(box)
                rows = max(rows, row + 1)
                break
            t.remove()
    ax.set_yticks(y, labels)
    ax.set_ylim(-0.6, n - 0.45 + 0.5 * rows)
    style(ax, xlabel)


def fig_loss():
    names = ['00_final', '03_loss_sqdist_plus_mi', '04_loss_l1', '01_loss_mi_smooth1', '02_loss_mi_smooth100']
    labels = ['squared distance (final, 20k)', 'squared distance + MI (12k)', 'mean distance, L1 (12k)',
              'MI, smoothness 1 (6k)', 'MI, smoothness 100 (6k)']
    s = {n: summary(n) for n in names}
    mi = json.loads((OUT / 'analysis' / 'mi_landscape_summary.json').read_text())
    curve = rows(OUT / 'analysis' / 'mi_landscape_slabs.csv')
    scales = [float(k[4:]) for k in curve[0] if k.startswith('mi_s')]
    med = [np.median([float(r[f'mi_s{x:g}']) - float(r['mi_s1']) for r in curve]) for x in scales]

    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 3.6), width_ratios=[1.6, 1], layout='constrained')
    bars(a, labels, [s[n]['tre'] for n in names], 0, 'median TRE, mm (1,080 real validation slabs, exact rigid)',
         references(s['00_final']))
    a.set_title('Loss functions')
    b.plot(scales, med, color=BLUE, linewidth=2, marker='o', markersize=5)
    b.axvline(1, color=MUTED, linewidth=0.9, linestyle=(0, (4, 3)))
    b.set_xticks([0, 0.5, 1, 1.5, 2], ['0\nno shift', '0.5', '1\ntruth', '1.5', '2'])
    b.set_xlabel(f"s: T1 moved by s × the true shift\nMI peaks at the truth on {mi['pct_peak_at_truth']:.0f} % "
                 f"of slabs, short of it on {mi['pct_peak_below_truth']:.0f} %,\nbeyond it on "
                 f"{mi['pct_peak_above_truth']:.0f} %")
    b.set_ylabel('median MI(s) − MI(truth)')
    b.set_title('MI along the true shift')
    b.grid(color=GRID, linewidth=0.8)
    for side in ('top', 'right'):
        b.spines[side].set_visible(False)
    return fig


def fig_output():
    names = ['00_final', '05_output_3d', '06_output_no_canonical', '07_output_direct_shift', '08_smooth_0',
             '09_smooth_1']
    labels = ['final: shift along PE, canonical,\nvelocity + scaling & squaring, smooth 0.1 (20k)',
              'free 3D shift (20k)', 'no canonical orientation (12k)', 'shift output directly,\nno scaling & squaring (12k)',
              'smoothness 0 (12k)', 'smoothness 1 (12k)']
    s = {n: summary(n) for n in names}
    fig, (a, b) = plt.subplots(1, 2, figsize=(12, 4.2), width_ratios=[1.6, 1], sharey=True, layout='constrained')
    bars(b, labels, [s[n]['fold_pct'] for n in names], 0, 'folded voxels, % (Jacobian ≤ 0)',
         fmt=lambda v: '0' if v == 0 else f'{v:.3f}')
    b.set_title('Folds')
    bars(a, labels, [s[n]['tre'] for n in names], 0, 'median TRE, mm (exact rigid)', references(s['00_final']))
    a.set_title('Output and regularisation')
    return fig


def fig_size():
    names = ['10_unet_half', '00_final', '11_unet_double']
    labels = ['×0.5 channels, 0.07 M (12k)', 'final, 0.29 M (20k)', '×2 channels, 1.16 M (12k)']
    s = {n: summary(n) for n in names}
    fig, a = plt.subplots(figsize=(8, 2.6), layout='constrained')
    bars(a, labels, [s[n]['tre'] for n in names], 1, 'median TRE, mm (exact rigid)', references(s['00_final']))
    a.set_title('U-Net width')
    return fig


def fig_rigid():
    sens = rows(OUT / 'analysis' / 'rigid_sensitivity_slabs.csv')
    errors = list(dict.fromkeys(r['rigid_error'] for r in sens))
    x = np.arange(len(errors))
    get = lambda model, key: [float(r[key]) for r in sens if r['model'] == model]
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(13, 3.6), width_ratios=[1, 1, 1.1], layout='constrained')
    for ax, key, title in [(a, 'shift_error', 'Shift alone (scored with the exact rigid)'),
                           (b, 'total_error', 'Total (scored with the given rigid)')]:
        if key == 'total_error':
            ax.plot(x, get('00_final', 'no_deformation'), color=GREY, linewidth=2, marker='o', label='no deformation')
        ax.plot(x, get('00_final', key), color=BLUE, linewidth=2, marker='o', label='final (trained with exact rigid)')
        ax.plot(x, get('12_rigid_noise', key), color=ORANGE, linewidth=2, marker='s',
                label='trained with rigid error ≤ 2 mm / 2°')
        ax.set_xticks(x, ['0' if e == '0-0' else e.replace('-', '–') for e in errors])
        ax.set_xlabel('rigid error added, mm and degrees')
        ax.set_ylabel('median error, mm')
        ax.set_title(title)
        ax.grid(color=GRID, linewidth=0.8)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
    a.legend(frameon=False, fontsize=8)
    b.legend(frameon=False, fontsize=8)
    s0, s12 = summary('00_final'), summary('12_rigid_noise')
    bars(c, ['rigid model alone', '+ final', '+ trained with rigid error'],
         [s0['no_deformation_rigid_model'], s0['tre_rigid_model'], s12['tre_rigid_model']], 1,
         'median TRE, mm', [('AFNI lpc+ZZ', s0['base_afni_lpc'])])
    c.set_title('On top of the rigid model')
    return fig


def fig_breakdown():
    r = rows(OUT / '00_final' / 'val_slabs.csv')
    groups = {'orientation': ['axial', 'coronal'],
              'thickness, mm': sorted({x['thickness'] for x in r}, key=float),
              'position, inferior → superior': sorted({x['position'] for x in r}, key=float)}
    key = {'orientation': 'orientation', 'thickness, mm': 'thickness', 'position, inferior → superior': 'position'}
    fmt = {'orientation': str, 'thickness, mm': str, 'position, inferior → superior': lambda v: f'{float(v):.0%}'}
    series = [('no deformation', 'no_deformation', GREY, 'o'), ('AFNI lpc+ZZ', 'base_afni_lpc', ORANGE, 'D'),
              ('final, on top of the rigid model', 'tre_rigid_model', AQUA, 's'),
              ('final, exact rigid', 'tre', BLUE, 'o')]
    items, ticks, y = [], [], 0.0
    for g, values in groups.items():
        y -= 0.6
        ticks.append((y, g, True))
        for v in values:
            y -= 1
            items.append((y, key[g], v))
            ticks.append((y, fmt[g](v), False))
        y -= 0.3
    fig, ax = plt.subplots(figsize=(9, 4.8), layout='constrained')
    for name, k, color, marker in series:
        med = [np.median([float(x[k]) for x in r if x[col] == v]) for _, col, v in items]
        ax.scatter(med, [i[0] for i in items], color=color, marker=marker, s=50, label=name, zorder=3,
                   edgecolor='white', linewidth=1.2)
    ax.set_yticks([t[0] for t in ticks], [t[1] for t in ticks])
    for label, (_, _, bold) in zip(ax.get_yticklabels(), ticks):
        label.set_fontweight('bold' if bold else 'normal')
        label.set_color(MUTED if bold else INK)
    ax.set_xlim(0, None)
    style(ax, 'median TRE, mm (1,080 real validation slabs)')
    for yi, _, _ in items:
        ax.axhline(yi, color=GRID, linewidth=0.5, zorder=0)
    ax.legend(frameon=False, fontsize=8, ncol=4, loc='lower left', bbox_to_anchor=(0, 1.0), handletextpad=0.2,
              columnspacing=1.2)
    return fig


def fig_architecture():
    """Block diagram with tensor sizes for an axial 16 mm slab (90 x 104 x 8 voxels of 2 mm, PE along the first axis)."""
    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.set_xlim(0, 15.5)
    ax.set_ylim(0, 4)
    ax.axis('off')
    boxes = [(1.15, 3.05, 2.2, 0.9, 'EPI slab\n90×104×8 voxels'),
             (1.15, 1.35, 2.2, 1.3, 'T1 resampled onto\nthe slab grid through\nthe rigid transform M'),
             (3.85, 2.2, 2.4, 2.1, 'stack: 2×90×104×8\n\ncanonical orientation\n(PE → axis Y, sign +)\n2×104×90×8\n'
                                    'pad: 2×112×96×16'),
             (6.8, 2.2, 2.6, 2.1, '3D U-Net, 0.29 M params\n\nencoder 16-32-32-32\n4 max-pools → 7×6×1\n'
                                  'decoder with skips\n→ 16×112×96×16'),
             (9.5, 2.2, 2.0, 2.1, 'velocity along PE\n1×112×96×16\n\nscaling & squaring\n(7 steps)'),
             (11.85, 2.2, 2.0, 2.1, 'shift u along PE\n\ncrop, back to the\nslab orientation\n90×104×8'),
             (14.3, 2.2, 2.3, 2.1, 'T1 position\nof voxel v\n\nM·H·(v + u(v)·e_PE)')]
    for i, (x, y, w, h, text) in enumerate(boxes):
        ax.add_patch(matplotlib.patches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.12', linewidth=1.2,
            facecolor='#e3eefb' if i == 3 else '#f4f3f0', edgecolor=BLUE if i == 3 else GRID))
        ax.text(x, y, text, ha='center', va='center', fontsize=9, color=INK, linespacing=1.35)
    arrow = dict(arrowstyle='-|>', color=MUTED, linewidth=1.2, shrinkA=0, shrinkB=0)
    ax.annotate('', (2.6, 2.75), (2.3, 3.05), arrowprops=arrow)
    ax.annotate('', (2.6, 1.65), (2.3, 1.35), arrowprops=arrow)
    for (x0, _, w0, _, _), (x1, _, w1, _, _) in zip(boxes[2:], boxes[3:]):
        ax.annotate('', (x1 - w1 / 2 - 0.05, 2.2), (x0 + w0 / 2 + 0.05, 2.2), arrowprops=arrow)
    ax.text(7.75, 0.35, 'Loss = mean squared distance (mm²) between the predicted and the true T1 position of the scored '
                        'slab voxels + 0.1 × smoothness of the velocity', ha='center', fontsize=9, color=MUTED)
    return fig


FIGURES = {'architecture': fig_architecture, 'loss': fig_loss, 'output': fig_output, 'size': fig_size,
           'rigid': fig_rigid, 'breakdown': fig_breakdown}


def upload(paths):
    """Upload figures as files of the W&B run `report_figures` (same run every time); {name: public URL}."""
    run = wandb.init(entity=ENTITY, project=PROJECT, id='report-figures', name='report_figures', job_type='report',
                     resume='allow', dir=FIG)
    for p in paths.values():
        run.save(str(p), base_path=str(p.parent), policy='now')
    run.finish()
    files = {f.name: f.url for f in wandb.Api().run(f'{ENTITY}/{PROJECT}/report-figures').files()}
    return {k: files[p.name] for k, p in paths.items()}


DESCRIPTION = ('After the rigid model has aligned an EPI slab to the T1 of the same subject, a small U-Net predicts how '
               'far each slab voxel is displaced along the phase-encoding (PE) axis by the susceptibility distortion. '
               '13 experiments, one seed each, scored on 1,080 real-distortion validation slabs.')


def section(heading, url, *lines):
    """Heading = the conclusion, one figure, a line of explanation."""
    return [wr.H1(heading), *[wr.P(t) for t in lines[:-1]], wr.Image(url), wr.P(lines[-1])]


def blocks(urls):
    s = {c: summary(c) for c in CONFIGS}
    t = {c: s[c]['tre'] for c in CONFIGS}
    f = s['00_final']
    mi = json.loads((OUT / 'analysis' / 'mi_landscape_summary.json').read_text())
    sens = rows(OUT / 'analysis' / 'rigid_sensitivity_slabs.csv')
    shift = [float(r['shift_error']) for r in sens if r['model'] == '00_final']
    return [
        wr.TableOfContents(),
        wr.H1('Final result: pending'),
        wr.CalloutBlock('The full chain (rigid model + this stage) on the 59 held-out test subjects (7,552 slabs) is '
                        'being scored by the team; the result will be added here. Everything below is validation: 1,080 '
                        'real-distortion slabs of 30 validation subjects.'),
        *section('Architecture: one shift per voxel along PE, from a 0.29 M-parameter U-Net', urls['architecture'],
                 'Input: the slab and the T1 resampled onto the slab grid through the rigid transform M. The slab is '
                 'flipped / transposed so that PE is always the same tensor axis with the same sign (canonical '
                 'orientation). The U-Net outputs a velocity along PE; scaling and squaring integrates it into a smooth '
                 'shift without folds. Tensor sizes are for an axial 16 mm slab.'),
        *section(f"Loss: squared distance to the true position {t['00_final']:.2f} mm; mutual information "
                 f"{min(t['01_loss_mi_smooth1'], t['02_loss_mi_smooth100']):.2f}–"
                 f"{max(t['01_loss_mi_smooth1'], t['02_loss_mi_smooth100']):.2f} mm", urls['loss'],
                 'Compared loss functions (each + λ × mean squared difference of the velocity between neighbouring '
                 'voxels, λ = 0.1 unless stated): squared distance = mean over the scored slab voxels of |predicted − '
                 'true T1 position|², in mm², the true position comes from the simulator; L1 = the same distance without '
                 'the square; MI = minus the mutual information between the slab and the T1 sampled at the predicted '
                 'positions, the classical similarity measure of multimodal registration, needs no ground truth '
                 '(λ = 1 and 100); squared distance + MI = both.',
                 f"Right: along the true shift, MI peaks at the truth on {mi['pct_peak_at_truth']:.0f} % of slabs and "
                 f"beyond it on {mi['pct_peak_above_truth']:.0f} %. TRE = mean distance between the predicted and the "
                 f"true T1 position over a slab; median over 1,080 slabs; the rigid transform is exact."),
        *section(f"Output: one shift along PE {t['00_final']:.2f} mm; free 3D shift {t['05_output_3d']:.2f} mm; "
                 f"without canonical orientation {t['06_output_no_canonical']:.2f} mm", urls['output'],
                 f"Without scaling and squaring: {t['07_output_direct_shift']:.2f} mm and "
                 f"{s['07_output_direct_shift']['fold_pct']:.3f} % folded voxels. Smoothness weight 0 / 0.1 / 1: "
                 f"{t['08_smooth_0']:.2f} / {t['00_final']:.2f} / {t['09_smooth_1']:.2f} mm."),
        *section(f"Network size: channels ×0.5 / ×1 / ×2 give {t['10_unet_half']:.2f} / {t['00_final']:.2f} / "
                 f"{t['11_unet_double']:.2f} mm", urls['size'],
                 '0.07 M / 0.29 M / 1.16 M parameters. The final model was trained for 20k steps, the other two for 12k.'),
        *section(f"Link to the rigid model: shift error {' / '.join(f'{v:.2f}' for v in shift)} mm at a rigid error of "
                 f"0 / 1–2 / 3–6 / 6–10 mm and degrees", urls['rigid'],
                 f"On top of the rigid model: {f['no_deformation_rigid_model']:.2f} mm without deformation, "
                 f"{f['tre_rigid_model']:.2f} mm with the final model, "
                 f"{s['12_rigid_noise']['tre_rigid_model']:.2f} mm with the model trained with random rigid errors of "
                 f"up to 2 mm and 2°. Rigid errors in the sensitivity test: random, the same for both models."),
        *section('Examples: slabs at the 10th, 50th and 90th percentile of TRE', urls['examples'],
                 'Yellow: T1 grey / white-matter boundary over the slab (middle slice) without and with the predicted '
                 'shift. Right: predicted and true shift along PE over the scored voxels, mm. Exact rigid transform.'),
        *section(f"By slab type: largest error on inferior slabs", urls['breakdown'],
                 'Medians over the slabs of each type. AFNI lpc+ZZ = the best classical baseline (affine, whole '
                 'pipeline); final on top of the rigid model = the realistic chain.'),
        wr.H1(f"Training curves: held-out simulated TRE {f['tre']:.2f} vs real"),
        curves(),
        wr.P('Each run: config, curves, summary (val/*), per-slab table (val_slabs), weights as the artifact '
             '<config>-model (best.pt + config.yaml).'),
        wr.H1('Data, speed, limitations'),
        wr.UnorderedList([
            'Data: HCP, 3 T. Training slabs simulated on the fly by slabsim from 199 training subjects (random oblique '
            'grid, head motion, PE axis and sign, distortion strength, header error); 64 fixed simulated slabs of 10 '
            'more training subjects choose the checkpoint; 1,080 real-distortion slabs of 30 validation subjects '
            '(measured field maps, whole-brain anchor) score the models.',
            'Speed: 0.29 M parameters, about 5 ms per slab for the network and scaling and squaring (RTX 5080); '
            'the T1 resampling is extra.',
            'Limitations: one seed per experiment; unequal budgets (6k / 12k / 20k steps); the ground truth itself is '
            'uncertain by 0.2–0.4 mm, so differences below ~0.3 mm are not claimed; simulated and real TRE differ; '
            'accuracy depends on the rigid stage.']),
    ]


def curves():
    """Training curves of all experiments, with the runs table."""
    lines = [('val_real_tre_120', 'Real validation TRE, 120 fixed slabs (not used for choosing)', 'median TRE, mm'),
             ('holdout_sim_tre', 'Held-out simulated TRE, 64 slabs (chooses the checkpoint)', 'median TRE, mm'),
             ('train_tre', 'Training TRE (mean over 100 steps)', 'mean TRE, mm'),
             ('loss', 'Training loss (scale differs between loss functions)', 'loss')]
    panels = [wr.LinePlot(title=t, x='Step', y=[k], title_y=ty, smoothing_factor=0.6 if k in ('train_tre', 'loss') else 0,
                          smoothing_type='exponential', legend_position='east',
                          layout=wr.Layout(x=12 * (i % 2), y=8 * (i // 2), w=12, h=8))
              for i, (k, t, ty) in enumerate(lines)]
    names = ', '.join(f"'{c}'" for c in CONFIGS)
    runs = wr.Runset(ENTITY, PROJECT, name='Experiments', filters=f'Name in [{names}]',
                     order=[wr.OrderBy('Name', ascending=True)],
                     visible_columns=['summary:val/tre', 'summary:val/pct_below_2mm', 'summary:val/fold_pct',
                                      'summary:val/tre_rigid_model', 'config:steps.value'])
    return wr.PanelGrid(runsets=[runs], panels=panels)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--url', help='existing report to overwrite')
    args = ap.parse_args()
    FIG.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, make in FIGURES.items():
        paths[name] = FIG / f'{name}.png'
        make().savefig(paths[name])
        plt.close('all')
    paths['examples'] = OUT / 'analysis' / 'examples.png'
    urls = upload(paths)
    report = wr.Report.from_url(args.url) if args.url else wr.Report(project=PROJECT, entity=ENTITY)
    report.title = 'DLIR partial-FOV: deformable stage'
    report.description = DESCRIPTION
    report.width = 'fixed'
    report.blocks = blocks(urls)
    report.save()
    print(report.url)


if __name__ == '__main__':
    main()
