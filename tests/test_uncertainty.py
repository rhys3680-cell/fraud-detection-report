import unittest
import numpy as np
from frauddetectionreport.robustness import paired_block_bootstrap


class BootstrapTests(unittest.TestCase):
    def test_constant_difference_has_exact_total(self):
        draws = paired_block_bootstrap(np.tile([2., -3.], (11, 1)), 3, 100)
        np.testing.assert_array_equal(draws, np.tile([22., -33.], (100, 1)))

    def test_pairs_share_resampled_days(self):
        x = np.arange(21.)
        draws = paired_block_bootstrap(np.column_stack([x, -x]), 7, 100)
        np.testing.assert_array_equal(draws[:, 0], -draws[:, 1])
        np.testing.assert_array_equal(draws, paired_block_bootstrap(np.column_stack([x, -x]), 7, 100))

    def test_full_length_circular_block_preserves_total(self):
        x = np.arange(15.).reshape(5, 3)
        draws = paired_block_bootstrap(x, 5, 100)
        np.testing.assert_array_equal(draws, np.tile(x.sum(axis=0), (100, 1)))

    def test_reject_invalid_block(self):
        with self.assertRaises(ValueError):
            paired_block_bootstrap(np.ones((5, 1)), 6)
