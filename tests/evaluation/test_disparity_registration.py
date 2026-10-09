"""Analytic numerical fixtures only; never street or sponsor execution evidence."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'scripts/evaluation'))
import disparity_registration as registration
import matched


def calibration(width=6, height=4, numerator=12, left_cx=3, right_cx=1):
    return (f'S_rect_101: {width} {height}\nS_rect_103: {width} {height}\n'
            f'P_rect_101: 10 0 {left_cx} 0 0 10 2 0 0 0 1 0\n'
            f'P_rect_103: 10 0 {right_cx} {-numerator} 0 10 2 0 0 0 1 0\n').encode()


class RegistrationTests(unittest.TestCase):
    def run_conversion(self, disparity=None, validity=None, source=None, original=(4, 6), crop=(2, 1, 2, 2), convention=registration.CONVENTION):
        if disparity is None:
            disparity = np.array([[4, 6], [8, 14]], dtype=np.float32)
        if validity is None:
            validity = np.ones((2, 2), dtype=np.uint8)
        return registration.register(disparity, validity, calibration() if source is None else source, original, crop, convention)

    def test_offset_and_nonorigin_embedding_full_denominator(self):
        depth, mask, record = self.run_conversion()
        expected = np.full((4, 6), np.nan, dtype=np.float32)
        expected[1:3, 2:4] = [[6, 3], [2, 1]]
        np.testing.assert_array_equal(depth, expected)
        self.assertEqual(mask.sum(), 4)
        self.assertEqual(record['outsideCropPixels'], 20)
        self.assertEqual(record['converterInputDimensions'], [2, 2])
        self.assertNotIn('inferenceOutputDimensions', record)
        counts, metrics, _ = matched.depth_scores(depth, np.ones((4, 6)), mask.astype(bool))
        self.assertEqual((counts['validReference'], counts['matchedPrediction'], counts['invalidPredictionOnReference']), (24, 4, 20))
        self.assertEqual(metrics['predictionCoverageOnFullReference'], 4/24)

    def test_source_scaling_and_edge_crop(self):
        depth, mask, record = self.run_conversion(source=calibration(width=12, height=8, numerator=24, left_cx=6, right_cx=2), crop=(4, 2, 2, 2))
        np.testing.assert_array_equal(depth[2:4, 4:6], [[6, 3], [2, 1]])
        self.assertEqual(record['sourceToDecodedScale'], [.5, .5])
        self.assertEqual(mask.sum(), 4)

    def test_nonfinite_nonpositive_and_source_invalid(self):
        disparity = np.array([[np.nan, np.inf], [2, 14]], dtype=np.float32)
        depth, mask, _ = self.run_conversion(disparity, np.array([[1, 1], [1, 0]], dtype=np.uint8))
        self.assertTrue(np.isnan(depth).all())
        self.assertFalse(mask.any())

    def test_negative_disparity_with_negative_offset(self):
        depth, mask, _ = self.run_conversion(np.full((2, 2), -1, dtype=np.float32), source=calibration(left_cx=1, right_cx=3))
        np.testing.assert_array_equal(depth[1:3, 2:4], np.full((2, 2), 12, dtype=np.float32))
        self.assertEqual(mask.sum(), 4)

    def test_float32_unrepresentable_depth_is_unsupported(self):
        for numerator, disparity in [(12, np.nextafter(np.float32(0), np.float32(1))), (1e-300, 1)]:
            with self.subTest(numerator=numerator):
                depth, mask, _ = self.run_conversion(np.full((2, 2), disparity, dtype=np.float32), source=calibration(numerator=numerator, left_cx=1, right_cx=1))
                self.assertTrue(np.isnan(depth).all())
                self.assertFalse(mask.any())

    def test_rejects_unsupported_inputs(self):
        for override in [dict(convention='right-minus-left'), dict(crop=(5, 1, 2, 2)), dict(crop=(2.0, 1, 2, 2)), dict(original=(True, 6)), dict(disparity=np.ones((2, 2), dtype=np.float64)), dict(disparity=np.ones((1, 4), dtype=np.float32)), dict(validity=np.full((2, 2), 2, dtype=np.uint8)), dict(source=calibration().replace(b'0 0 1 0', b'0 1 1 0'))]:
            with self.subTest(override=override), self.assertRaises(ValueError):
                self.run_conversion(**override)


if __name__ == '__main__':
    unittest.main()
