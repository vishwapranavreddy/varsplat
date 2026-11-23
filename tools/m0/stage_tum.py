"""Extract a verified TUM backup only inside the allocated job's scratch directory."""
import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import tarfile


def stage(archive, target):
    scratch = Path(os.environ['TMPDIR']).resolve()
    target = target.resolve()
    if not target.is_relative_to(scratch) or target == scratch:
        raise ValueError('Extraction target must be strictly inside job scratch')
    h = hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda: f.read(1024**2), b''): h.update(b)
    expected = 'e983d6830916e66dc4a46a71368046b149b283de87769690e7aa4e0b9483530c'
    if h.hexdigest() != expected: raise ValueError('Archive differs from verified cleanup backup')
    target.mkdir(exist_ok=False)
    total = 0
    with tarfile.open(archive, 'r:gz') as tf:
        for member in tf:
            parts = PurePosixPath(member.name).parts
            if not parts or parts[0] != 'rgbd_dataset_freiburg1_desk' or '..' in parts or member.issym() or member.islnk():
                raise ValueError('Unexpected archive path/type: ' + member.name)
            dest = target.joinpath(*parts[1:])
            if not dest.resolve().is_relative_to(target): raise ValueError('Path escape')
            if member.isdir(): dest.mkdir(exist_ok=True, parents=True)
            elif member.isfile():
                total += member.size
                if total > 2_000_000_000: raise ValueError('Expanded input exceeds reservation')
                dest.parent.mkdir(exist_ok=True, parents=True)
                with tf.extractfile(member) as source, dest.open('xb') as out:
                    shutil.copyfileobj(source, out)
            else: raise ValueError('Unexpected archive member')
    counts = {}
    for kind in ('rgb', 'depth'):
        paths = [target / line.split()[1] for line in (target / (kind + '.txt')).read_text().splitlines() if line.strip() and not line.lstrip().startswith('#')]
        for path in paths:
            if not path.resolve().is_relative_to(target): raise ValueError('Unsafe image path')
            with path.open('rb') as f:
                if f.read(8) != b'\x89PNG\r\n\x1a\n': raise ValueError('Missing/invalid PNG: ' + str(path))
        counts[kind] = len(paths)
    return {'status': 'PASSED', 'archive_sha256': expected, 'expanded_bytes': total, 'listed_images': counts}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('archive', type=Path)
    p.add_argument('target', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    report = stage(args.archive, args.target)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report))
