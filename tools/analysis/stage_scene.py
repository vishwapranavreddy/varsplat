"""Stage pinned public TUM/Replica archives in allocated scratch, one scene at a time."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import tarfile
import zipfile

TUM_SHA256 = {
    'rgbd_dataset_freiburg1_desk': 'e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c',
    'rgbd_dataset_freiburg1_desk2': 'a569e4cb453a3cd9285bc985fcb109e65f055c75b33a4b155acd9a68d96b77d2',
    'rgbd_dataset_freiburg1_room': '5ace47a1d2e53696bc939a84999293a04a7226958e848a609666691fd3cc38da',
    'rgbd_dataset_freiburg2_xyz': 'a4903027a7fe6bc6573a1ce96a8afa5689f15d74f4a3f0627e91b5799f62e40b',
    'rgbd_dataset_freiburg3_long_office_household': 'c7cd8e1afb87c80e5744a356214819b110fa09b4744fa4ba0cc2382f9ba59e9c',
}
REPLICA_SHA256 = 'dc18265e213f8281444b768eed76f627bde8c3b925654934f84f78a87990a548'
REPLICA_SCENES = ('room0', 'room1', 'room2', 'office0', 'office1', 'office2', 'office3', 'office4')


def safe_destination(target, name, prefix):
    path = PurePosixPath(name)
    parts = path.parts
    if path.is_absolute() or '..' in parts or '\\' in name or tuple(parts[:len(prefix)]) != tuple(prefix):
        raise ValueError('Unexpected archive path: ' + name)
    dest = target.joinpath(*parts[len(prefix):])
    if not dest.resolve().is_relative_to(target.resolve()):
        raise ValueError('Path escape: ' + name)
    return dest


def stage(archive, target, dataset, scene):
    if dataset == 'tum_rgbd':
        expected = TUM_SHA256[scene]
        limit = 10_000_000_000
    elif dataset == 'replica' and scene in REPLICA_SCENES:
        expected = REPLICA_SHA256
        limit = 4_000_000_000
    else:
        raise ValueError('Unlisted dataset/scene')
    scratch = Path(os.environ['TMPDIR']).resolve()
    target = target.resolve()
    if not target.is_relative_to(scratch) or target == scratch:
        raise ValueError('Destination must be strictly inside job scratch')
    digest = hashlib.sha256()
    with archive.open('rb') as f:
        for chunk in iter(lambda: f.read(1024**2), b''):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise ValueError('Archive checksum differs from the pinned recovery receipt')
    target.mkdir(exist_ok=False)
    total, files = 0, 0
    if dataset == 'tum_rgbd':
        with tarfile.open(archive, 'r:gz') as tf:
            for member in tf:
                dest = safe_destination(target, member.name, (scene,))
                if member.isdir():
                    dest.mkdir(parents=True, exist_ok=True)
                elif member.isfile():
                    total += member.size
                    if total > limit: raise ValueError('Expanded input exceeds reservation')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with tf.extractfile(member) as source, dest.open('xb') as output:
                        shutil.copyfileobj(source, output)
                    files += 1
                else:
                    raise ValueError('Unsupported archive member type')
        counts = {}
        for kind in ('rgb', 'depth'):
            paths = [line.split()[1] for line in (target / (kind + '.txt')).read_text().splitlines()
                     if line.strip() and not line.lstrip().startswith('#')]
            if not paths: raise ValueError('Empty image list')
            for name in paths:
                dest = safe_destination(target, name, ())
                with dest.open('rb') as f:
                    if f.read(8) != b'\x89PNG\r\n\x1a\n': raise ValueError('Invalid/missing PNG: ' + name)
            counts[kind] = len(paths)
        if not (target / 'groundtruth.txt').is_file(): raise ValueError('Missing TUM ground truth')
    else:
        prefix = ('Replica', scene)
        with zipfile.ZipFile(archive) as zf:
            for member in zf.infolist():
                if tuple(PurePosixPath(member.filename).parts[:2]) != prefix:
                    continue
                dest = safe_destination(target, member.filename, prefix)
                if stat.S_ISLNK(member.external_attr >> 16): raise ValueError('Archive contains a symlink')
                if member.is_dir():
                    dest.mkdir(parents=True, exist_ok=True)
                else:
                    total += member.file_size
                    if total > limit: raise ValueError('Expanded input exceeds reservation')
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    with zf.open(member) as source, dest.open('xb') as output:
                        shutil.copyfileobj(source, output)  # Also verifies ZIP CRC.
                    files += 1
        counts = {'rgb': 0, 'depth': 0}
        for i in range(2000):
            for kind, name, signature in [('rgb', f'frame{i:06d}.jpg', b'\xff\xd8'),
                                          ('depth', f'depth{i:06d}.png', b'\x89PNG\r\n\x1a\n')]:
                with (target / 'results' / name).open('rb') as f:
                    if f.read(len(signature)) != signature: raise ValueError('Invalid image: ' + name)
                counts[kind] += 1
        if len((target / 'traj.txt').read_text().splitlines()) != 2000:
            raise ValueError('Expected 2000 Replica poses')
        if files != 4001: raise ValueError('Unexpected Replica scene file count')
    return {'status': 'PASSED', 'dataset': dataset, 'scene': scene, 'archive_sha256': expected,
            'expanded_bytes': total, 'files': files, 'listed_images': counts}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['tum_rgbd', 'replica'], required=True)
    parser.add_argument('--scene', required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    with args.output.open('x') as f:
        report = stage(args.archive, args.target, args.dataset, args.scene)
        json.dump(report, f, indent=2)
    print(json.dumps(report))
