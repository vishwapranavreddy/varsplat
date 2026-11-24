import hashlib
import os
from pathlib import Path
import tarfile
import tempfile
import unittest
from unittest.mock import patch

from tools.analysis.stage_scene import safe_destination, stage, TUM_SHA256


class StageTests(unittest.TestCase):
    def test_member_path_boundaries(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root)
            self.assertEqual(safe_destination(target, 'scene/rgb/a.png', ('scene',)), target / 'rgb/a.png')
            for name in ('/scene/a', 'scene/../escape', 'different/a', 'scene\\escape'):
                with self.subTest(name=name), self.assertRaises(ValueError):
                    safe_destination(target, name, ('scene',))

    def test_hash_and_scratch_guards_precede_extraction(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / 'archive.tgz'
            archive.write_bytes(b'wrong archive')
            with patch.dict(os.environ, TMPDIR=str(root)):
                with self.assertRaisesRegex(ValueError, 'strictly inside'):
                    stage(archive, root, 'tum_rgbd', 'rgbd_dataset_freiburg1_desk')
                with self.assertRaisesRegex(ValueError, 'checksum'):
                    stage(archive, root / 'input', 'tum_rgbd', 'rgbd_dataset_freiburg1_desk')
            self.assertFalse((root / 'input').exists())

    def test_tar_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as root:
            root = Path(root)
            archive = root / 'archive.tgz'
            with tarfile.open(archive, 'w:gz') as f:
                item = tarfile.TarInfo('fixture/link')
                item.type = tarfile.SYMTYPE
                item.linkname = '../outside'
                f.addfile(item)
            digest = hashlib.sha256(archive.read_bytes()).hexdigest()
            with patch.dict(os.environ, TMPDIR=str(root)), patch.dict(TUM_SHA256, fixture=digest):
                with self.assertRaisesRegex(ValueError, 'member type'):
                    stage(archive, root / 'input', 'tum_rgbd', 'fixture')
            self.assertFalse((root / 'outside').exists())


if __name__ == '__main__': unittest.main()
