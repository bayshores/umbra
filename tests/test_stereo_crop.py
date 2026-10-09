"""Analytic crop tests only; tiny synthetic inputs are never runtime assets."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
spec = importlib.util.spec_from_file_location('stereo_crop', ROOT / 'scripts/prepare-stereo-crop.py')
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


class StereoCropTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix='stereo-crop-test-')
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.rgb = np.arange(6 * 8 * 3, dtype=np.uint8).reshape(6, 8, 3)
        self.right = 255 - self.rgb
        self.raw = np.full((6, 8), 16, dtype=np.uint16)
        self.raw[1, 3] = 0
        self.raw[2, 5] = 32
        self.fields = {
            'S_rect_101': [16, 12], 'S_rect_103': [16, 12],
            'P_rect_101': [8, 0, 7.5, 0, 0, 10, 5.5, 0, 0, 0, 1, 0],
            'P_rect_103': [8, 0, 5.5, -8, 0, 10, 5.5, 0, 0, 0, 1, 0],
        }
        self.args = argparse.Namespace(
            left=self.root / 'source-left.png', right=self.root / 'source-right.png',
            depth=self.root / 'source-depth.png', calibration=self.root / 'source-calibration.txt',
            crop=[2, 1, 4, 3], depth_divisor=4., output=self.root / 'output', expected_disparity=True)
        Image.fromarray(self.rgb).save(self.args.left)
        Image.fromarray(self.right).save(self.args.right)
        Image.fromarray(self.raw).save(self.args.depth)
        self.write_calibration()

    def write_calibration(self):
        self.args.calibration.write_text(''.join(
            f'{key}: {" ".join(map(str, values))}\n' for key, values in self.fields.items()))

    def test_exact_pixels_geometry_and_provenance(self):
        manifest = adapter.prepare(self.args)
        for name, source in (('left', self.rgb), ('right', self.right), ('depth', self.raw)):
            with Image.open(self.args.output / f'{name}.png') as image:
                np.testing.assert_array_equal(np.array(image), source[1:4, 2:6])
                self.assertEqual(image.mode, 'I;16' if name == 'depth' else 'RGB')
        calibration = json.loads((self.args.output / 'calibration.json').read_text())
        transform = np.array(manifest['crop']['source_to_crop_matrix'])
        point = np.array([.7, -.2, 4., 1.])
        projections = []
        for side in ('left', 'right'):
            original = np.array(calibration['original'][f'{side}_projection'])
            cropped = np.array(calibration['transformed'][f'{side}_projection'])
            np.testing.assert_array_equal(cropped, transform @ original)
            source_uv = (original @ point)[:2] / (original @ point)[2]
            crop_uv = (cropped @ point)[:2] / (cropped @ point)[2]
            np.testing.assert_allclose(crop_uv, source_uv - [2, 1], rtol=0, atol=1e-15)
            np.testing.assert_array_equal(cropped[[0, 1, 0], [0, 1, 3]],
                                          original[[0, 1, 0], [0, 1, 3]])
            projections.append((source_uv, crop_uv))
        self.assertAlmostEqual(projections[0][0][0] - projections[1][0][0], 2.)
        self.assertAlmostEqual(projections[0][1][0] - projections[1][1][0], 2.)
        self.assertEqual(calibration['source_to_decoded_scale'], [.5, .5])
        self.assertEqual(calibration['original']['left_projection'][0][0], 4.)
        self.assertEqual(calibration['principal_point_offset_px'], 1.)
        self.assertEqual(calibration['implied_baseline_m'], 1.)
        expected = np.load(self.args.output / 'expected-disparity.npy', allow_pickle=False)
        self.assertTrue(np.isnan(expected[0, 1]))
        self.assertEqual(expected[1, 3], 1.5)
        self.assertEqual(np.count_nonzero(np.isfinite(expected)), 11)
        support = manifest['support']
        self.assertEqual(support['source_nonzero_depth'], 47)
        self.assertEqual(support['crop_nonzero_depth'], 11)
        self.assertEqual(support['projected_right_u_outside_crop'], 5)
        self.assertEqual(support['projected_right_u_inside_crop'], 6)
        self.assertEqual(support['projected_right_u_within_one_pixel_of_crop_border'], 3)
        for name, item in manifest['sources'].items():
            self.assertEqual(item['sha256'], hashlib.sha256(Path(item['path']).read_bytes()).hexdigest())
        for name, item in manifest['outputs'].items():
            self.assertEqual(item['sha256'], hashlib.sha256((self.args.output / name).read_bytes()).hexdigest())

    def test_invalid_controls_reject_without_output(self):
        for crop in ([0, 0, 0, 1], [-1, 0, 1, 1], [0, 0, 9, 6], [7, 5, 2, 1],
                     [0, 0, 2**40, 1], [False, 0, 1, 1], [0, .5, 1, 1]):
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                adapter.prepare(argparse.Namespace(**{**vars(self.args), 'crop': crop}))
            self.assertFalse(self.args.output.exists())
        for divisor in (0, -1, float('nan'), float('inf'), 1e-308):
            with self.subTest(divisor=divisor), self.assertRaises((ValueError, FloatingPointError)):
                adapter.prepare(argparse.Namespace(**{**vars(self.args), 'depth_divisor': divisor}))
            self.assertFalse(self.args.output.exists())
        self.raw[1, 2] = 1
        Image.fromarray(self.raw).save(self.args.depth)
        with self.assertRaises(FloatingPointError):
            adapter.prepare(argparse.Namespace(**{**vars(self.args), 'depth_divisor': 1e308}))
        self.assertFalse(self.args.output.exists())

    def test_calibration_uses_verified_snapshot_during_transient_change(self):
        original = self.args.calibration.read_bytes()
        parse = adapter.rectified_calibration.parse_rectified_pair
        changed = False

        def transient_change(text, width, height):
            nonlocal changed
            changed = True
            altered = original.decode().replace('8 0', '12 0')
            self.args.calibration.write_text(altered)
            try:
                return parse(text, width, height)
            finally:
                self.args.calibration.write_bytes(original)

        with patch.object(adapter.rectified_calibration, 'parse_rectified_pair',
                          side_effect=transient_change):
            manifest = adapter.prepare(self.args)
        self.assertTrue(changed)
        calibration = json.loads((self.args.output / 'calibration.json').read_text())
        self.assertEqual(calibration['original']['left_projection'][0][0], 4.)
        self.assertEqual(calibration['original']['right_projection'][0][0], 4.)
        self.assertEqual(manifest['sources']['calibration']['sha256'], hashlib.sha256(original).hexdigest())
        self.assertEqual(self.args.calibration.read_bytes(), original)

    def test_format_and_dimension_admission(self):
        cases = [('right', np.zeros((6, 8), dtype=np.uint8)),
                 ('right', np.zeros((5, 8, 3), dtype=np.uint8)),
                 ('depth', np.zeros((6, 8), dtype=np.uint8)),
                 ('depth', np.zeros((6, 8, 3), dtype=np.uint8)),
                 ('depth', np.zeros((5, 8), dtype=np.uint16))]
        for name, invalid in cases:
            with self.subTest(name=name, shape=invalid.shape, dtype=invalid.dtype):
                original = getattr(self.args, name).read_bytes()
                Image.fromarray(invalid).save(getattr(self.args, name))
                with self.assertRaises(ValueError):
                    adapter.prepare(self.args)
                self.assertFalse(self.args.output.exists())
                getattr(self.args, name).write_bytes(original)

    def test_existing_output_and_staging_failure_are_preserved(self):
        self.args.output.mkdir()
        sentinel = self.args.output / 'keep.txt'
        sentinel.write_text('retained')
        with self.assertRaises(ValueError):
            adapter.prepare(self.args)
        self.assertEqual(sentinel.read_text(), 'retained')
        sentinel.unlink()
        staged = self.root / 'concurrent-stage'
        staged.mkdir()
        (staged / 'data').write_text('staged')
        with self.assertRaises(OSError):
            adapter.publish_new_directory(staged, self.args.output)
        self.assertEqual(list(self.args.output.iterdir()), [])
        self.assertEqual((staged / 'data').read_text(), 'staged')
        self.args.output.rmdir()
        with patch.object(Image.Image, 'save', side_effect=OSError('staging failure')):
            with self.assertRaises(OSError):
                adapter.prepare(self.args)
        self.assertFalse(self.args.output.exists())
        self.assertEqual(list(self.root.glob('.stereo-crop-*')), [])

    def test_calibration_restrictions_reused_and_optional_disparity(self):
        for index in (5, 6, 7):
            value = self.fields['P_rect_103'][index]
            self.fields['P_rect_103'][index] += 1
            self.write_calibration()
            with self.subTest(right_projection_index=index), self.assertRaises(ValueError):
                adapter.prepare(self.args)
            self.assertFalse(self.args.output.exists())
            self.fields['P_rect_103'][index] = value
        self.write_calibration()
        self.args.expected_disparity = False
        adapter.prepare(self.args)
        self.assertFalse((self.args.output / 'expected-disparity.npy').exists())

    def test_cli_requires_explicit_divisor(self):
        command = [sys.executable, '-B', str(ROOT / 'scripts/prepare-stereo-crop.py')]
        for name in ('left', 'right', 'depth', 'calibration', 'output'):
            command.extend(['--' + name, str(getattr(self.args, name))])
        command.extend(['--crop', '2', '1', '4', '3'])
        result = subprocess.run(command, capture_output=True, text=True, timeout=30)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn('--depth-divisor', result.stderr)
        self.assertFalse(self.args.output.exists())


if __name__ == '__main__':
    unittest.main()
