import unittest

from utils.group_allfixed_features import (
    ALLFIXED_LOCAL_FEATURES,
    ALLFIXED_VARIANTS,
    QUOTA_V1_FIXED_AUX_CONTROL,
    QUOTA_V2_ALL_FIXED,
)
from utils.group_quota_v2 import GROUP_FAMILY_QUOTA64_V2_FEATURES
from utils.power_curve import GROUP_TURBINE_PREFIXES
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


class GroupAllFixedFeaturesTest(unittest.TestCase):
    def test_variants_have_identical_input_counts(self) -> None:
        self.assertEqual(
            ALLFIXED_VARIANTS,
            (QUOTA_V1_FIXED_AUX_CONTROL, QUOTA_V2_ALL_FIXED),
        )
        self.assertEqual(len(ALLFIXED_LOCAL_FEATURES), 6)

        expected = {
            "kpx_group_1": 100,
            "kpx_group_2": 100,
            "kpx_group_3": 94,
        }
        for group, turbines in GROUP_TURBINE_PREFIXES.items():
            v1_count = len(GROUP_FAMILY_QUOTA65_V1_FEATURES[group]) + len(
                turbines
            ) * len(ALLFIXED_LOCAL_FEATURES)
            v2_count = len(GROUP_FAMILY_QUOTA64_V2_FEATURES[group]) + len(
                turbines
            ) * len(ALLFIXED_LOCAL_FEATURES)
            self.assertEqual(v1_count, expected[group])
            self.assertEqual(v2_count, expected[group])


if __name__ == "__main__":
    unittest.main()
