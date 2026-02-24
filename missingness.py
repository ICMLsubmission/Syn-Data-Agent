from typing import List
import numpy as np
import pandas as pd


def apply_missingness(df: pd.DataFrame, key_cols: List[str], missing_field_rate: float) -> pd.DataFrame:
    """
    Randomly set non-key fields to missing at approximately missing_field_rate.
    Does not touch key columns.
    """
    if df is None or len(df) == 0:
        return df

    if missing_field_rate <= 0:
        return df

    out = df.copy()
    cols = [c for c in out.columns if c not in set(key_cols)]
    if not cols:
        return out

    # Apply per-cell missingness
    mask = np.random.rand(len(out), len(cols)) < missing_field_rate
    for j, c in enumerate(cols):
        # Do not null out columns that are entirely non-null keys (already excluded), so safe here.
        out.loc[mask[:, j], c] = np.nan

    return out
