import unittest

import numpy as np

from classicalBaselines.trainClassicalBaselines import (
    extract_features,
    feature_names,
    generate_random_seeds,
    stratified_cap,
)


class ClassicalBaselineTests(unittest.TestCase):
    def test_feature_schema_is_finite_and_channel_major(self):
        rng = np.random.default_rng(29)
        windows = rng.normal(size=(12, 128, 6)).astype(np.float32)
        features = extract_features(windows)

        self.assertEqual(features.shape, (12, 126))
        self.assertEqual(len(feature_names(6)), features.shape[1])
        self.assertTrue(np.isfinite(features).all())

    def test_master_seed_generation_is_reproducible(self):
        expected = [1591207480, 1814784297, 230512677]
        self.assertEqual(generate_random_seeds(17, 3), expected)
        self.assertEqual(generate_random_seeds(17, 3), expected)

    def test_stratified_cap_is_deterministic_and_balanced(self):
        x = np.arange(300, dtype=np.float32).reshape(100, 3)
        y = np.repeat(np.arange(4), 25)
        capped_x, capped_y, capped = stratified_cap(x, y, 40, seed=29)
        repeated_x, repeated_y, _ = stratified_cap(x, y, 40, seed=29)

        self.assertTrue(capped)
        self.assertEqual(len(capped_y), 40)
        self.assertEqual(np.bincount(capped_y).tolist(), [10, 10, 10, 10])
        np.testing.assert_array_equal(capped_x, repeated_x)
        np.testing.assert_array_equal(capped_y, repeated_y)


if __name__ == "__main__":
    unittest.main()
