from typing import Dict, List, Tuple
import pandas as pd


def basic_report(
    visits: List[int],
    dm: pd.DataFrame,
    mh: pd.DataFrame,
    vs: pd.DataFrame,
    lb: pd.DataFrame,
    ae: pd.DataFrame,
) -> Dict:
    report = {}

    report["row_counts"] = {
        "DM": int(len(dm)),
        "MH": int(len(mh)) if mh is not None else 0,
        "VS": int(len(vs)) if vs is not None else 0,
        "LB": int(len(lb)) if lb is not None else 0,
        "AE": int(len(ae)) if ae is not None else 0,
    }

    n_subj = int(len(dm))
    report["n_subjects"] = n_subj

    # Completion rate per visit (VS/LB presence)
    def completion(df: pd.DataFrame, label: str) -> Dict[str, float]:
        if df is None or len(df) == 0:
            return {str(v): 0.0 for v in visits}
        out = {}
        for v in visits:
            n = df[df["VISITNUM"] == v]["USUBJID"].nunique()
            out[str(v)] = float(n) / float(n_subj) if n_subj else 0.0
        return out

    report["vs_completion_by_visitnum"] = completion(vs, "VS")
    report["lb_completion_by_visitnum"] = completion(lb, "LB")

    # Missingness per table (% cells missing)
    def missingness(df: pd.DataFrame) -> float:
        if df is None or len(df) == 0:
            return 0.0
        total = df.size
        miss = int(df.isna().sum().sum())
        return float(miss) / float(total) if total else 0.0

    report["missingness_fraction"] = {
        "DM": missingness(dm),
        "MH": missingness(mh) if mh is not None else 0.0,
        "VS": missingness(vs) if vs is not None else 0.0,
        "LB": missingness(lb) if lb is not None else 0.0,
        "AE": missingness(ae) if ae is not None else 0.0,
    }

    # AE severity distribution
    if ae is not None and len(ae) > 0 and "AESEV" in ae.columns:
        sev = ae["AESEV"].value_counts(dropna=False).to_dict()
    else:
        sev = {}
    report["ae_severity_counts"] = {str(k): int(v) for k, v in sev.items()}

    return report
