from __future__ import annotations

from utils.per_turbine_features import LOCAL_TREE_FEATURES
from utils.tree_feature_profiles import GROUP_FAMILY_QUOTA65_V1_FEATURES


SOURCES = ("ldaps", "gfs")


def raw_source_input_columns(group: str, source: str) -> list[str]:
    if source not in SOURCES:
        raise ValueError(f"Unknown weather source: {source}")
    other = "gfs" if source == "ldaps" else "ldaps"
    candidates = [
        *GROUP_FAMILY_QUOTA65_V1_FEATURES[group],
        *LOCAL_TREE_FEATURES,
    ]
    columns = []
    for column in candidates:
        name = column.lower()
        if other in name or name.startswith("wake_"):
            continue
        if column not in columns:
            columns.append(column)
    if not columns:
        raise ValueError(f"No raw source features selected for {group} {source}")
    return columns
