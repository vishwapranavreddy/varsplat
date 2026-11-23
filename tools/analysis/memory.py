"""Filter nvidia-smi samples by exact PID and SLAM interval, retaining raw evidence."""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path


def summarize(report, text, offset_seconds):
    if not report.get('slam_complete'):
        raise ValueError('A completed SLAM interval is required')
    start, end = report['slam_started_unix'], report['slam_finished_unix']
    if end <= start: raise ValueError('Invalid SLAM interval')
    tz = timezone(timedelta(seconds=offset_seconds))
    samples = []
    for row in csv.DictReader(text.splitlines(), skipinitialspace=True):
        if row.get('pid') == 'pid': continue  # Repeated nvidia-smi header.
        try:
            pid = int(row['pid'])
        except (ValueError, TypeError): continue
        if pid != report['pid']: continue
        timestamp = datetime.strptime(row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=tz).timestamp()
        if not start <= timestamp <= end: continue
        value = row['used_gpu_memory [MiB]'].strip().split()[0]
        if not value.isdigit(): raise ValueError('Memory unavailable for target process')
        samples.append((timestamp, row['gpu_uuid'], int(value) * 1024**2))
    if len(samples) < 2: raise ValueError('Insufficient matching process samples')
    devices = {s[1] for s in samples}
    if len(devices) != 1: raise ValueError('This single-GPU experiment unexpectedly spans devices')
    samples.sort()
    gaps = [b[0]-a[0] for a, b in zip(samples, samples[1:])]
    if samples[0][0]-start > 2 or end-samples[-1][0] > 2 or max(gaps) > 2:
        raise ValueError('Memory sampling has an unexplained coverage gap')
    return {'pid': report['pid'], 'gpu_uuid': next(iter(devices)), 'sample_count': len(samples),
            'sampled_slam_max_process_bytes': max(s[2] for s in samples),
            'maximum_sample_gap_seconds': max(gaps), 'utc_offset_seconds': offset_seconds,
            'scope': 'driver-reported process memory sampled at approximately 500 ms; not an exact peak',
            'slam_started_unix': start, 'slam_finished_unix': end}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--report', required=True, type=Path)
    p.add_argument('--samples', required=True, type=Path)
    p.add_argument('--utc-offset-seconds', required=True, type=int)
    p.add_argument('--output', required=True, type=Path)
    args = p.parse_args()
    result = summarize(json.loads(args.report.read_text()), args.samples.read_text(), args.utc_offset_seconds)
    result['report_sha256'] = hashlib.sha256(args.report.read_bytes()).hexdigest()
    result['samples_sha256'] = hashlib.sha256(args.samples.read_bytes()).hexdigest()
    with args.output.open('x') as f: json.dump(result, f, indent=2)
