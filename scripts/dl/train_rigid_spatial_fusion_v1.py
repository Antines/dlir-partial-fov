#!/usr/bin/env python
"""Train the 6-DOF rigid stage with simulator-derived geometric supervision."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
import yaml

HERE = Path(__file__).resolve()
REPO = HERE.parents[2]
sys.path.insert(0, str(REPO / 'scripts'))

from slabsim.dataset import usable_runs
from dl.rigid_data import SavedSlabDataset, SlabStream
from dl.rigid_eval import evaluate_model
from dl.rigid_losses import geometric_tre_loss
from dl.rigid_model_spatial_fusion_v1 import RigidRegistrationNet


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default=str(REPO / 'configs' / 'rigid_spatial_fusion_easy_v1.yaml'))
    p.add_argument('--data', default=os.environ.get('DLIR_DATA', str(REPO / 'dlir_data')))
    p.add_argument('--output', default=None)
    p.add_argument('--resume', default=None)
    return p.parse_args()


def load_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def split(name):
    return (REPO / 'splits' / f'hcp_{name}.txt').read_text().split()


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def maybe_wandb(cfg, out_dir):
    wb = cfg.get('logging', {}).get('wandb', {})
    if not wb.get('enabled', False):
        raise RuntimeError('W&B must be enabled for tracked CNN-baseline training')
    import wandb
    wandb.login()
    identity_file = out_dir / 'wandb_run_id.txt'
    run_id = identity_file.read_text().strip() if identity_file.is_file() else wandb.util.generate_id()
    run = wandb.init(project=wb.get('project', 'neuroml-dlir'), name=wb.get('name'),
                     config=cfg, dir=str(out_dir), id=run_id, resume='allow', mode='online')
    identity_file.write_text(run.id + '\n')
    return run

def main():
    args = parse_args()
    cfg = load_yaml(args.config)
    seed_all(int(cfg['seed']))
    data_root = Path(args.data)
    out_dir = Path(args.output or cfg['output_dir'])
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / 'config.yaml').write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics_file = out_dir / 'metrics.jsonl'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)
    runs = usable_runs(data_root / 'data', split('train'))
    print('training runs:', len(runs))

    dcfg = cfg['data']
    stream = SlabStream(
        runs,
        seed=cfg['seed'],
        slabs_per_run=dcfg['slabs_per_run'],
        difficulty=dcfg['difficulty'],
        probe_points=dcfg['probe_points'],
    )
    loader = DataLoader(stream, batch_size=None, num_workers=dcfg['num_workers'], persistent_workers=dcfg['num_workers'] > 0)
    it = iter(loader)
    val = SavedSlabDataset(data_root, max_slabs=dcfg.get('val_max_slabs'))
    print('validation slabs:', len(val))

    mcfg = cfg['model']
    model = RigidRegistrationNet(
        grid_shape_xyz=mcfg['grid_shape_xyz'],
        base_channels=mcfg['base_channels'],
        max_rotation_deg=mcfg['max_rotation_deg'],
        max_translation_mm=mcfg['max_translation_mm'],
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(cfg['optim']['lr']), weight_decay=float(cfg['optim']['weight_decay']))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, int(cfg['train']['epochs'])))
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda' and cfg['train'].get('amp', True)))

    start_epoch, global_step, best = 0, 0, float('inf')
    if args.resume:
        ck = torch.load(args.resume, map_location=device, weights_only=True)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer'])
        if 'scheduler' in ck: scheduler.load_state_dict(ck['scheduler'])
        start_epoch = ck['epoch'] + 1; global_step = ck.get('global_step', 0); best = ck.get('best_val_tre', best)
        print('resumed:', args.resume, 'next epoch', start_epoch)

    wb = maybe_wandb(cfg, out_dir)
    steps_per_epoch = int(cfg['train']['steps_per_epoch'])
    log_every = int(cfg['train']['log_every'])

    for epoch in range(start_epoch, int(cfg['train']['epochs'])):
        model.train(); losses = []; t0 = time.time()
        for step in range(steps_per_epoch):
            batch = next(it)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda' and cfg['train'].get('amp', True))):
                out = model(batch['slab'], batch['slab_affine'], batch['t1'], batch['t1_affine'])
                pts = batch['probe_points'].to(device=device, dtype=torch.float32)
                tgt = batch['target_probe_points'].to(device=device, dtype=torch.float32)
                loss = geometric_tre_loss(out.matrix, pts, tgt)
            scaler.scale(loss).backward()
            if cfg['train'].get('grad_clip'):
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(cfg['train']['grad_clip']))
            scaler.step(opt); scaler.update()
            global_step += 1; losses.append(float(loss.detach()))
            if global_step % log_every == 0:
                rec = {'epoch': epoch, 'step': global_step, 'train_tre_loss_mm': float(np.mean(losses[-log_every:])), 'lr': opt.param_groups[0]['lr']}
                print(rec)
                with metrics_file.open('a') as f: f.write(json.dumps(rec) + '\n')
                if wb: wb.log(rec, step=global_step)

        val_summary, _ = evaluate_model(model, val, device)
        rec = {'epoch': epoch, 'step': global_step, 'train_epoch_loss_mm': float(np.mean(losses)), **{f'val_{k}': v for k, v in val_summary.items()}, 'epoch_s': time.time() - t0}
        print(json.dumps(rec, indent=2))
        with metrics_file.open('a') as f: f.write(json.dumps(rec) + '\n')
        if wb: wb.log(rec, step=global_step)

        scheduler.step()
        current = val_summary['median_tre_mm']
        is_best = current < best
        best = min(best, current)
        state = {
            'epoch': epoch, 'global_step': global_step, 'best_val_tre': best,
            'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scheduler': scheduler.state_dict(),
            'config': cfg,
        }
        torch.save(state, out_dir / 'latest.pt')
        if is_best:
            torch.save(state, out_dir / 'best.pt')

    if wb: wb.finish()


if __name__ == '__main__':
    main()
