from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np


@dataclass(frozen=True)
class VisitDef:
    visit: str
    visitnum: int  # weeks from baseline


def build_subject_visit_plan(
    usubjids: List[str],
    visits: Tuple[VisitDef, ...],
    dropout_rate: float,
    missed_visit_rate: float,
) -> Dict[str, Dict]:
    """
    Returns per-subject plan:
      {
        "last_visitnum": int,
        "completed_visitnums": set(int),
        "missed_visitnums": set(int)
      }
    Rules:
    - With probability dropout_rate, subject drops after a random visit (can be baseline or later).
    - Visits > last_visitnum are not generated.
    - Among visits <= last_visitnum, each non-baseline visit can be missed with probability missed_visit_rate.
      (Baseline is always present.)
    """
    visitnums = [v.visitnum for v in visits]
    max_vn = max(visitnums)

    plan = {}
    for sid in usubjids:
        is_dropout = np.random.rand() < dropout_rate

        if is_dropout:
            # Choose last completed visit among available visitnums
            last_visitnum = int(np.random.choice(visitnums))
        else:
            last_visitnum = int(max_vn)

        eligible = [vn for vn in visitnums if vn <= last_visitnum]
        eligible_set = set(eligible)

        missed = set()
        for vn in eligible:
            if vn == 0:
                continue  # baseline always present
            if np.random.rand() < missed_visit_rate:
                missed.add(vn)

        completed = eligible_set - missed
        if 0 not in completed:
            completed.add(0)
            missed.discard(0)

        plan[sid] = {
            "last_visitnum": last_visitnum,
            "completed_visitnums": completed,
            "missed_visitnums": missed,
        }

    return plan
