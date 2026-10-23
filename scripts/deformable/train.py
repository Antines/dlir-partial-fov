"""Train one experiment of the deformable stage (a YAML file in configs/) and score it.

Run from scripts/:  python -m deformable.train deformable/configs/00_final.yaml [--rigid CSV]
- Each step: one random training slab simulated on the fly by slabsim (199 training subjects), Adam, gradient norm
  clipped at 5. Loss terms: see configs/00_final.yaml.
- Every 1,000 steps: median TRE on 64 simulated slabs of 10 held-out training subjects (chooses the checkpoint) and on
  120 fixed real validation slabs (shows the simulation-to-real gap; never used for choosing).
- At the end the chosen checkpoint is scored on all 1,080 validation slabs (evaluate.py).
Outputs: runs/deformable/<config name>/ (config.yaml, best.pt, last.pt, val_slabs.csv, val_summary.json) and a W&B
run in the project dlir-deformable: curves, summary, per-slab table, the weights as an artifact.
"""
import argparse
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
from torch.utils.data import DataLoader, Subset

from .data import REPO, TrainStream, ValSlabs, holdout_set, read_rigid, split_runs
from .evaluate import evaluate, save
from .losses import mutual_information, smoothness, voxel_errors
from .model import build, device, predict, warp

OUT = REPO / 'runs' / 'deformable'


def step_loss(net, cfg, b, gen):
    """Loss of one training slab, and values to log."""
    u, v = predict(net, cfg, b['x'], b['pe_axis'], b['pe_sign'])
    err = voxel_errors(u, b, b['M'])
    loss = cfg['smooth'] * smoothness(v)
    if cfg['loss'] in ('sqdist', 'sqdist+mi'):
        loss = loss + err.pow(2).mean()
    if cfg['loss'] == 'l1':
        loss = loss + err.mean()
    mi = torch.zeros(())
    if cfg['loss'] in ('mi', 'sqdist+mi'):
        x = b['x'][None].to(u.device)
        weight = (b['t1mask'].to(u.device) & (x[0, 0] > 0.05)).float()[None, None]   # brain voxels with EPI signal
        mi = mutual_information(x[:, :1], warp(x[:, 1:], u, padding='border'), weight, gen=gen)
        loss = loss - mi
    return loss, {'loss': loss.item(), 'train_tre': err.mean().item(), 'train_mi': mi.item(),
                  'max_shift_vox': u.abs().max().item()}


@torch.no_grad()
def median_tre(net, cfg, slabs):
    net.eval()
    tre = [voxel_errors(predict(net, cfg, b['x'], b['pe_axis'], b['pe_sign'])[0], b, b['M']).mean().item()
           for b in slabs]
    net.train()
    return float(np.median(tre))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('config')
    ap.add_argument('--rigid', help="CSV with the rigid model's transform of every validation slab (data.read_rigid)")
    args = ap.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    name = Path(args.config).stem
    out = OUT / name
    out.mkdir(parents=True, exist_ok=True)
    (out / 'config.yaml').write_text(Path(args.config).read_text())
    dev = device()
    torch.manual_seed(cfg['seed'])
    gen = torch.Generator().manual_seed(cfg['seed'])
    run = wandb.init(project='dlir-deformable', name=name, config=cfg, dir=out)

    train_runs, hold_runs = split_runs()
    hold = holdout_set(hold_runs, cache=OUT / 'holdout.pt')
    real = Subset(ValSlabs(), np.random.default_rng(0).choice(1080, 120, replace=False).tolist())
    net = build(cfg).to(dev)
    opt = torch.optim.Adam(net.parameters(), lr=cfg['lr'])
    loader = DataLoader(TrainStream(train_runs, cfg['seed'], cfg['rigid_noise']), batch_size=None, num_workers=3,
                        prefetch_factor=4, persistent_workers=True)
    best, logs = float('inf'), []
    for step, b in enumerate(loader):
        if step % 1000 == 0 or step == cfg['steps']:
            h = median_tre(net, cfg, hold)
            r = median_tre(net, cfg, DataLoader(real, batch_size=None, num_workers=2))
            wandb.log({'holdout_sim_tre': h, 'val_real_tre_120': r}, step=step)
            print(f'step {step}: held-out simulated TRE {h:.3f} mm, real validation TRE {r:.3f} mm', flush=True)
            if h < best:
                best = h
                torch.save(net.state_dict(), out / 'best.pt')
        if step == cfg['steps']:
            break
        loss, values = step_loss(net, cfg, b, gen)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        logs.append(values)
        if (step + 1) % 100 == 0:
            wandb.log({k: float(np.mean([v[k] for v in logs])) for k in logs[0]}, step=step + 1)
            logs = []
    torch.save(net.state_dict(), out / 'last.pt')

    net.load_state_dict(torch.load(out / 'best.pt', map_location=dev))
    rows, summary = evaluate(net, cfg, rigid=read_rigid(args.rigid) if args.rigid else None)
    save(rows, summary, out, 'val')
    run.summary.update({f'val/{k}': v for k, v in summary.items()})
    run.log({'val_slabs': wandb.Table(columns=list(rows[0]), data=[list(r.values()) for r in rows])})
    model = wandb.Artifact(f'{name}-model', type='model')
    model.add_file(str(out / 'best.pt'))
    model.add_file(str(out / 'config.yaml'))
    run.log_artifact(model)
    run.finish()


if __name__ == '__main__':
    main()
