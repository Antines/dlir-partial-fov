"""Download the HCP-YA 3T files needed by the slab simulator (S1200 raw data on S3, AWS profile 'hcp').

Per subject: the scan-metadata CSV, the raw T1w and, for every rest run, its SBRef and spin-echo pair. Also the
SBRef + spin-echo pair of a 'bridge' task run acquired in the same session, shim group and PE direction (for the
bridged-BBR baseline). For the first --gdc subjects, also HCP's gradient-corrected T1w_acpc_dc_restore
(to measure gradient nonlinearity).
Subjects are taken in a seeded random order; the first --n complete ones are listed in subjects.txt.

Usage: python download_hcp.py <out_dir> [--n 300] [--seed 0] [--gdc 10] [--workers 8]
"""
import argparse
import csv
import io
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config
from botocore.exceptions import ClientError

BUCKET, PREFIX = 'hcp-openaccess', 'HCP_1200/'
RUNS = ['rfMRI_REST1_LR', 'rfMRI_REST1_RL', 'rfMRI_REST2_LR', 'rfMRI_REST2_RL']
TRANSFER = TransferConfig(max_concurrency=4)


def list_subjects(s3):
    pages = s3.get_paginator('list_objects_v2').paginate(Bucket=BUCKET, Prefix=PREFIX, Delimiter='/')
    return sorted(p[len(PREFIX):-1] for p in pages.search('CommonPrefixes[].Prefix') if p)


def fetch(s3, key, dst):
    """Download one object unless it is already there; False if it does not exist on S3."""
    if dst.exists():
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        s3.download_file(BUCKET, key, str(dst), Config=TRANSFER)
    except ClientError as e:
        if e.response['Error']['Code'] in ('404', 'NoSuchKey'):
            return False
        raise
    return True


def bridges(csv_file):
    """For every rest run: the first task run with the same session, shim group and PE direction."""
    rows = list(csv.DictReader(io.StringIO(csv_file.read_text())))
    same = lambda a, b: all(a[k] == b[k] for k in ('Session', 'Shim Group', 'PE Direction'))
    out = {}
    for r in rows:
        if r['Scan Type'] == 'rfMRI_SBRef':
            tasks = [t['Scan Description'].removesuffix('_SBRef') for t in rows
                     if t['Scan Type'] == 'tfMRI_SBRef' and same(t, r)]
            if tasks:
                out[r['Scan Description'].removesuffix('_SBRef')] = tasks[0]
    return out


def run_files(s3, subject, run, base, local):
    names = [f'{subject}_3T_{run}_SBRef.nii.gz', f'{subject}_3T_SpinEchoFieldMap_LR.nii.gz',
             f'{subject}_3T_SpinEchoFieldMap_RL.nii.gz']
    return all([fetch(s3, f'{base}/{run}/{n}', local / run / n) for n in names])


def download_subject(s3, subject, out, gdc):
    """All files of one subject; returns a status dict (complete = T1 + all four rest runs)."""
    base, local = f'{PREFIX}{subject}/unprocessed/3T', out / subject / 'unprocessed' / '3T'
    status = {'csv': fetch(s3, f'{base}/{subject}_3T.csv', local / f'{subject}_3T.csv'),
              'T1w': fetch(s3, f'{base}/T1w_MPR1/{subject}_3T_T1w_MPR1.nii.gz',
                           local / 'T1w_MPR1' / f'{subject}_3T_T1w_MPR1.nii.gz')}
    status.update({run: run_files(s3, subject, run, base, local) for run in RUNS})
    status['bridges'] = bridges(local / f'{subject}_3T.csv') if status['csv'] else {}
    for run, task in status['bridges'].items():
        status[f'bridge_{run}'] = run_files(s3, subject, task, base, local)
    if gdc:
        status['T1w_acpc_dc'] = fetch(s3, f'{PREFIX}{subject}/T1w/T1w_acpc_dc_restore.nii.gz',
                                      out / subject / 'T1w' / 'T1w_acpc_dc_restore.nii.gz')
    status['complete'] = status['T1w'] and all(status[r] for r in RUNS)
    return status


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--n', type=int, default=300)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gdc', type=int, default=10)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    config = Config(retries={'total_max_attempts': 8, 'mode': 'adaptive'}, max_pool_connections=64)
    s3 = boto3.Session(profile_name='hcp', region_name='us-east-1').client('s3', config=config)
    try:
        subjects = list_subjects(s3)
        random.Random(args.seed).shuffle(subjects)
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / 'subjects_ranked.txt').write_text('\n'.join(subjects) + '\n')
        candidates = subjects[:int(args.n * 1.1) + 5]               # margin for incomplete subjects
        with ThreadPoolExecutor(args.workers) as pool:
            status = dict(zip(candidates, pool.map(
                lambda ix: download_subject(s3, ix[1], args.out, ix[0] < args.gdc), enumerate(candidates))))
    except ClientError as e:
        print(f'Error: {e}', file=sys.stderr)
        return 1
    complete = [s for s in candidates if status[s]['complete']][:args.n]
    (args.out / 'subjects.txt').write_text('\n'.join(complete) + '\n')
    (args.out / 'download_status.json').write_text(json.dumps(status, indent=1))
    print(f'{len(complete)} complete subjects of {len(candidates)} tried; '
          f'{sum(1 for s in complete if len(status[s]["bridges"]) == 4)} with bridges for all four rest runs')
    return 0


if __name__ == '__main__':
    sys.exit(main())
