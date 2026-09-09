import unittest

import numpy as np
import pandas as pd

from frauddetectionreport.features import add_uid, build_features


class FeatureTests(unittest.TestCase):
    def test_uid_requires_address(self):
        df = pd.DataFrame(
            {
                "TransactionDT": [86_400, 86_401],
                "TransactionAmt": [10.0, 20.0],
                "card1": [1.0, 1.0],
                "addr1": [np.nan, np.nan],
                "D1": [1.0, 1.0],
            }
        )

        out = add_uid(df)

        self.assertEqual(out["uid_is_estimated"].sum(), 0)
        self.assertEqual(out["uid"].nunique(), 2)

    def test_training_aggregates_only_use_prior_rows(self):
        df = pd.DataFrame(
            {
                "TransactionID": [1, 2, 3],
                "isFraud": [0, 0, 0],
                "TransactionDT": [86_400, 172_800, 259_200],
                "TransactionAmt": [10.0, 20.0, 30.0],
                "card1": [1.0, 1.0, 1.0],
                "addr1": [100.0, 100.0, 100.0],
                "D1": [0.0, 1.0, 2.0],
                "DeviceInfo": ["x", "x", "x"],
            }
        )

        out, _ = build_features(df, fit_mask=np.array([True, True, False]))

        self.assertTrue(np.isnan(out.loc[0, "card1_amt_mean"]))
        self.assertEqual(out.loc[1, "card1_amt_mean"], 10.0)
        self.assertEqual(out.loc[2, "card1_amt_mean"], 15.0)


if __name__ == "__main__":
    unittest.main()
