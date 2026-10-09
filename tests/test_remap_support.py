"""Bounded analytic checks; synthetic masks do not establish physical support."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from remap_support import supported_rectangle


def exhaustive_rectangle(mask, crop_size, margin):
    """Independent oracle: enumerate every outer rectangle, without histograms."""
    candidates = []
    for top in range(mask.shape[0]):
        for bottom in range(top + 1, mask.shape[0] + 1):
            for left in range(mask.shape[1]):
                for right in range(left + 1, mask.shape[1] + 1):
                    width, height = right - left - 2 * margin, bottom - top - 2 * margin
                    if width >= crop_size[0] and height >= crop_size[1] and mask[top:bottom, left:right].all():
                        rectangle = (left + margin, top + margin, width, height)
                        candidates.append(((-width * height, top + margin, left + margin, height, width), rectangle))
    return min(candidates)[1] if candidates else None


class RemapSupportTests(unittest.TestCase):
    def compare_oracle(self, mask, crop_size, margin):
        before = mask.copy()
        expected = exhaustive_rectangle(mask, crop_size, margin)
        if expected is None:
            with self.assertRaises(ValueError):
                supported_rectangle(mask, crop_size, margin)
        else:
            actual = supported_rectangle(mask, crop_size, margin)
            self.assertEqual(actual, expected)
            self.assertTrue(all(type(value) is int for value in actual))
            x, y, width, height = actual
            self.assertTrue(mask[y - margin:y + height + margin, x - margin:x + width + margin].all())
        np.testing.assert_array_equal(mask, before)

    def test_all_3_by_3_masks_against_exhaustive_oracle(self):
        for bits in range(1 << 9):
            mask = np.array([bool(bits & (1 << i)) for i in range(9)]).reshape(3, 3)
            for crop, margin in [((1, 1), 0), ((2, 2), 0), ((2, 1), 0), ((1, 1), 1)]:
                with self.subTest(bits=bits, crop=crop, margin=margin):
                    self.compare_oracle(mask, crop, margin)

    def test_holes_disjoint_regions_margins_and_crop_fit(self):
        holes = np.ones((6, 7), bool); holes[2:4, 3] = False
        disjoint = np.zeros((8, 10), bool); disjoint[0:4, 0:4] = True; disjoint[3:8, 6:10] = True
        cases = [(holes, (2, 2), 0), (holes, (1, 1), 1), (disjoint, (2, 2), 1),
                 (np.ones((7, 8), bool), (3, 2), 2), (np.ones((3, 5), bool), (4, 2), 1),
                 (np.zeros((4, 5), bool), (1, 1), 0)]
        for mask, crop, margin in cases:
            with self.subTest(shape=mask.shape, crop=crop, margin=margin):
                self.compare_oracle(mask, crop, margin)

    def test_area_is_scored_after_removing_margin(self):
        mask = np.zeros((9, 19), bool)
        mask[0:3, 0:12] = True  # Outer area 36, interior area 10.
        mask[4:9, 13:19] = True  # Outer area 30, interior area 12.
        self.assertEqual(supported_rectangle(mask, (1, 1), 1), (14, 5, 4, 3))
        self.compare_oracle(mask, (1, 1), 1)

    def test_exact_top_left_height_width_ties(self):
        top_first = np.zeros((5, 7), bool); top_first[0:2, 4:7] = True; top_first[3:5, 0:3] = True
        self.assertEqual(supported_rectangle(top_first, (1, 1), 0), (4, 0, 3, 2))
        left_first = np.zeros((2, 7), bool); left_first[:, 0:3] = True; left_first[:, 4:7] = True
        self.assertEqual(supported_rectangle(left_first, (1, 1), 0), (0, 0, 3, 2))
        height_first = np.ones((3, 6), bool); height_first[2, 4:] = False
        self.assertEqual(supported_rectangle(height_first, (1, 1), 0), (0, 0, 6, 2))
        for mask in [top_first, left_first, height_first]:
            self.compare_oracle(mask, (1, 1), 0)

    def test_identical_common_mask_copies_and_layouts(self):
        left = np.ones((8, 11), bool); left[0, :] = False; left[4, 6] = False
        right = np.ones_like(left); right[:, 0] = False; right[5:, 8:] = False
        common = left & right
        expected = supported_rectangle(common, (2, 2), 1)
        expanded = np.zeros((8, 22), bool); expanded[:, ::2] = common
        readonly = common.copy(); readonly.flags.writeable = False
        for mask in [common.copy(), np.asfortranarray(common), expanded[:, ::2], readonly]:
            self.assertEqual(supported_rectangle(mask, (2, 2), 1), expected)
            self.compare_oracle(mask, (2, 2), 1)

    def test_numpy_integer_controls(self):
        self.assertEqual(supported_rectangle(np.ones((5, 6), bool), np.array([2, 2]), np.int64(1)), (1, 1, 4, 3))
        self.assertEqual(supported_rectangle(np.ones((1, 1), bool), [np.uint64(1), np.int32(1)], 0), (0, 0, 1, 1))

    def test_malformed_inputs_raise_value_error(self):
        masks = [None, [[True]], np.ones((2, 2), np.uint8), np.ones((2, 2), float),
                 np.array([[True]], object), np.ones(2, bool), np.ones((1, 1, 1), bool),
                 np.zeros((0, 3), bool), np.zeros((3, 0), bool), np.ma.array([[True]], mask=[[True]])]
        for mask in masks:
            with self.subTest(mask_type=type(mask)), self.assertRaises(ValueError):
                supported_rectangle(mask, (1, 1), 0)
        mask = np.ones((5, 6), bool)
        for crop in [None, (), (1,), (1, 1, 1), '11', {1: 2, 2: 3}, {1, 2}, np.array(1),
                     np.array([[1, 1]]), (0, 1), (-1, 1), (True, 1), (np.bool_(True), 1),
                     (1.0, 1), ('1', 1), (np.nan, 1), (np.inf, 1)]:
            with self.subTest(crop=crop), self.assertRaises(ValueError):
                supported_rectangle(mask, crop, 0)
        for margin in [None, -1, True, np.bool_(False), 1.0, np.nan, np.inf, '1', [], np.array(1)]:
            with self.subTest(margin=margin), self.assertRaises(ValueError):
                supported_rectangle(mask, (1, 1), margin)


if __name__ == '__main__':
    unittest.main()
