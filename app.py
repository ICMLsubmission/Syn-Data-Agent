import io
import json
import zipfile
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
from faker import Faker


# ---------------------------
# Config (v1: no LLM parsing)
# ---------------------------

@dataclass
class VisitDef:
    visit: str
    visitnum: int  # weeks from baseline


@dataclass
class GenConfig:
    studyid: str = "RA-P2-DEMO"
    disease: str = "Rheumatoid Arthritis"
    phase: str = "Phase 2"
    n_subjects: int = 100
    n_sites: int = 5
    arms: Tuple[str, str] = ("PLACEBO", "TRT")
    arm_ratio: Tuple[float, float] = (0.5, 0.5)
    visits: Tuple[VisitDef, ...] = (
        VisitDef("BASELINE", 0),
        VisitDef("WK2", 2),
        VisitDef("WK4", 4),
        VisitDef("WK6", 6),
        VisitDef("WK8", 8),
    )
    severe_ae_rate: float = 0.20  # among AEs
    # NOTE: v1 uses simple defaults. Later LLM will overwrite these.
    baseline_window_days: int = 60
    visit_jitter_days: int = 3
    ae_mean_per_subject: float = 0.6  # Poisson mean (0.6 => many subjects have 0 AEs)


# ---------------------------
# Helpers
# ---------------------------

def _set_seed(seed: int):
    np.random.seed(seed)


def _clamp(x: float, lo: float, hi: float) -> float:
    return float(min(max(x, lo), hi))


def _today() -> date:
    return datetime.utcnow().date()


def _rand_baseline_date(cfg: GenConfig) -> date:
    # Baseline within last cfg.baseline_window_days
    offset = np.random.randint(0, cfg.baseline_window_days + 1)
    return _today() - timedelta(days=int(offset))


def _visit_date(baseline: date, weeks: int, jitter_days: int) -> date:
    target = baseline + timedelta(days=int(weeks * 7))
    jitter = np.random.randint(-jitter_days, jitter_days + 1)
    return target + timedelta(days=int(jitter))


def _pick_weighted(items: List[str], probs: List[float]) -> str:
    return str(np.random.choice(items, p=np.array(probs) / np.sum(probs)))


# ---------------------------
# Table generators
# ---------------------------

def generate_dm(cfg: GenConfig, fake: Faker) -> pd.DataFrame:
    # Sites
    site_ids = [f"S{str(i+1).zfill(3)}" for i in range(cfg.n_sites)]

    # Subjects
    usubjid = [f"RA-{str(i+1).zfill(4)}" for i in range(cfg.n_subjects)]
    site_for_subj = np.random.choice(site_ids, size=cfg.n_subjects, replace=True)

    # Arms assignment
    arm = np.random.choice(list(cfg.arms), size=cfg.n_subjects, p=cfg.arm_ratio)

    sexes = ["M", "F"]
    races = ["ASIAN", "WHITE", "BLACK", "OTHER"]
    race_probs = [0.45, 0.35, 0.10, 0.10]

    countries = ["IND", "USA", "GBR", "DEU", "FRA", "CAN", "AUS"]
    country_probs = [0.50, 0.12, 0.08, 0.08, 0.07, 0.07, 0.08]

    rows = []
    for i in range(cfg.n_subjects):
        age = int(np.random.randint(18, 76))  # typical adult RA trial
        randdt = _rand_baseline_date(cfg)

        rows.append(
            {
                "STUDYID": cfg.studyid,
                "SITEID": site_for_subj[i],
                "USUBJID": usubjid[i],
                "ARM": arm[i],
                "RANDDT": randdt.isoformat(),
                "SEX": np.random.choice(sexes),
                "AGE": age,
                "RACE": _pick_weighted(races, race_probs),
                "COUNTRY": _pick_weighted(countries, country_probs),
            }
        )

    dm = pd.DataFrame(rows)
    return dm


def generate_mh(cfg: GenConfig, dm: pd.DataFrame) -> pd.DataFrame:
    # 0–3 MH rows per subject
    mh_terms = [
        "Hypertension",
        "Type 2 Diabetes Mellitus",
        "Hyperlipidemia",
        "Osteoporosis",
        "Asthma",
        "Depression",
        "Hypothyroidism",
        "Gastroesophageal reflux disease",
    ]
    rows = []
    mhid_counter = 1

    for _, r in dm.iterrows():
        k = int(np.random.randint(0, 4))  # 0..3
        randdt = datetime.fromisoformat(r["RANDDT"]).date()
        for _ in range(k):
            term = np.random.choice(mh_terms)
            # start date before baseline (up to 10 years)
            back_days = int(np.random.randint(30, 3650))
            mhstdtc = (randdt - timedelta(days=back_days)).isoformat()
            ongoing = np.random.choice(["Y", "N"], p=[0.75, 0.25])

            rows.append(
                {
                    "STUDYID": cfg.studyid,
                    "USUBJID": r["USUBJID"],
                    "MHID": f"MH-{str(mhid_counter).zfill(6)}",
                    "MHTERM": term,
                    "MHSTDTC": mhstdtc,
                    "MHONGO": ongoing,
                }
            )
            mhid_counter += 1

    return pd.DataFrame(rows)


def _baseline_vs_profile(sex: str, age: int) -> Dict[str, float]:
    # Loose plausible baselines
    weight = np.random.normal(72, 14)
    weight = _clamp(weight, 40, 120)

    sysbp = np.random.normal(125, 15)
    diast = np.random.normal(78, 10)
    hr = np.random.normal(78, 12)
    temp = np.random.normal(36.8, 0.3)

    return {
        "WEIGHT_KG": _clamp(weight, 40, 120),
        "SYSBP": _clamp(sysbp, 90, 170),
        "DIABP": _clamp(diast, 55, 110),
        "HR": _clamp(hr, 50, 120),
        "TEMP_C": _clamp(temp, 36.0, 39.0),
    }


def generate_vs(cfg: GenConfig, dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in dm.iterrows():
        baseline = datetime.fromisoformat(r["RANDDT"]).date()
        prof = _baseline_vs_profile(r["SEX"], int(r["AGE"]))

        # small drift + noise across visits
        for v in cfg.visits:
            vdt = _visit_date(baseline, v.visitnum, cfg.visit_jitter_days)

            # weight: small random walk
            prof["WEIGHT_KG"] = _clamp(prof["WEIGHT_KG"] + np.random.normal(0, 0.6), 40, 120)
            prof["SYSBP"] = _clamp(prof["SYSBP"] + np.random.normal(0, 2.5), 90, 170)
            prof["DIABP"] = _clamp(prof["DIABP"] + np.random.normal(0, 2.0), 55, 110)
            prof["HR"] = _clamp(prof["HR"] + np.random.normal(0, 2.5), 50, 120)
            prof["TEMP_C"] = _clamp(prof["TEMP_C"] + np.random.normal(0, 0.08), 36.0, 39.0)

            rows.append(
                {
                    "STUDYID": cfg.studyid,
                    "USUBJID": r["USUBJID"],
                    "VISIT": v.visit,
                    "VISITNUM": v.visitnum,
                    "VISITDT": vdt.isoformat(),
                    "SYSBP": round(prof["SYSBP"], 1),
                    "DIABP": round(prof["DIABP"], 1),
                    "HR": round(prof["HR"], 1),
                    "TEMP_C": round(prof["TEMP_C"], 2),
                    "WEIGHT_KG": round(prof["WEIGHT_KG"], 1),
                }
            )

    vs = pd.DataFrame(rows)
    return vs


def _baseline_lb_profile(arm: str) -> Dict[str, float]:
    # RA-ish inflammation elevated at baseline; treatment can slightly improve across visits (optional)
    crp = _clamp(np.random.lognormal(mean=np.log(8), sigma=0.55), 0.2, 60.0)  # mg/L-ish
    esr = _clamp(np.random.normal(35, 18), 2, 120)  # mm/hr-ish

    alt = _clamp(np.random.normal(25, 10), 5, 120)
    ast = _clamp(np.random.normal(23, 9), 5, 120)
    hgb = _clamp(np.random.normal(13.2, 1.4), 8.0, 18.0)
    wbc = _clamp(np.random.normal(6.8, 1.8), 2.5, 16.0)
    plt = _clamp(np.random.normal(290, 70), 100, 600)

    return {"CRP": crp, "ESR": esr, "ALT": alt, "AST": ast, "HGB": hgb, "WBC": wbc, "PLT": plt}


def generate_lb(cfg: GenConfig, dm: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in dm.iterrows():
        baseline = datetime.fromisoformat(r["RANDDT"]).date()
        arm = r["ARM"]
        prof = _baseline_lb_profile(arm)

        for idx, v in enumerate(cfg.visits):
            vdt = _visit_date(baseline, v.visitnum, cfg.visit_jitter_days)

            # Optional simple "treatment improves inflammation" flavor:
            # treatment: CRP/ESR drift down slightly over time; placebo: flat/noisy
            if idx > 0:
                if arm == "TRT":
                    prof["CRP"] = _clamp(prof["CRP"] * np.random.uniform(0.88, 0.98), 0.1, 60.0)
                    prof["ESR"] = _clamp(prof["ESR"] + np.random.normal(-2.0, 4.0), 2, 120)
                else:
                    prof["CRP"] = _clamp(prof["CRP"] * np.random.uniform(0.95, 1.05), 0.1, 60.0)
                    prof["ESR"] = _clamp(prof["ESR"] + np.random.normal(0.0, 5.0), 2, 120)

            # Liver enzymes mostly stable, occasional bumps
            prof["ALT"] = _clamp(prof["ALT"] + np.random.normal(0, 3.5), 5, 180)
            prof["AST"] = _clamp(prof["AST"] + np.random.normal(0, 3.0), 5, 180)

            prof["HGB"] = _clamp(prof["HGB"] + np.random.normal(0, 0.2), 8.0, 18.0)
            prof["WBC"] = _clamp(prof["WBC"] + np.random.normal(0, 0.4), 2.5, 18.0)
            prof["PLT"] = _clamp(prof["PLT"] + np.random.normal(0, 10), 100, 700)

            rows.append(
                {
                    "STUDYID": cfg.studyid,
                    "USUBJID": r["USUBJID"],
                    "VISIT": v.visit,
                    "VISITNUM": v.visitnum,
                    "VISITDT": vdt.isoformat(),
                    "CRP": round(prof["CRP"], 2),
                    "ESR": round(prof["ESR"], 1),
                    "ALT": round(prof["ALT"], 1),
                    "AST": round(prof["AST"], 1),
                    "HGB": round(prof["HGB"], 2),
                    "WBC": round(prof["WBC"], 2),
                    "PLT": int(round(prof["PLT"], 0)),
                }
            )

    lb = pd.DataFrame(rows)
    return lb


def generate_ae(cfg: GenConfig, dm: pd.DataFrame) -> pd.DataFrame:
    ae_terms = [
        "Headache",
        "Nausea",
        "Upper respiratory tract infection",
        "Injection site reaction",
        "Rash",
        "Diarrhea",
        "Elevated ALT",
        "Urinary tract infection",
        "Dizziness",
        "Fatigue",
    ]
    rel_terms = ["RELATED", "NOT RELATED"]
    rel_probs = [0.55, 0.45]

    rows = []
    aeid_counter = 1

    for _, r in dm.iterrows():
        baseline = datetime.fromisoformat(r["RANDDT"]).date()
        last_visit = _visit_date(baseline, cfg.visits[-1].visitnum, cfg.visit_jitter_days)
        study_end = last_visit + timedelta(days=7)

        n_ae = int(np.random.poisson(cfg.ae_mean_per_subject))
        # Cap for demo readability
        n_ae = min(n_ae, 4)

        for _ in range(n_ae):
            term = np.random.choice(ae_terms)

            # Start any time in study window
            total_days = max((study_end - baseline).days, 1)
            start_offset = int(np.random.randint(0, total_days))
            aestdt = baseline + timedelta(days=start_offset)

            # Duration 1-14 days typical
            dur = int(np.random.randint(1, 15))
            aeendt = min(aestdt + timedelta(days=dur), study_end)

            # Severity
            sev = _pick_weighted(["MILD", "MODERATE", "SEVERE"], [0.55, 0.25, cfg.severe_ae_rate])
            # Normalize so severe rate is "about" cfg.severe_ae_rate by using weighted pick:
            # (not exact; good enough for demo)

            aeser = "Y" if sev == "SEVERE" else _pick_weighted(["Y", "N"], [0.05, 0.95])
            aere = _pick_weighted(rel_terms, rel_probs)

            rows.append(
                {
                    "STUDYID": cfg.studyid,
                    "USUBJID": r["USUBJID"],
                    "AEID": f"AE-{str(aeid_counter).zfill(6)}",
                    "AETERM": term,
                    "AESTDTC": aestdt.isoformat(),
                    "AEENDTC": aeendt.isoformat(),
                    "AESEV": sev,
                    "AESER": aeser if sev != "SEVERE" else "Y",
                    "AEREL": aere,
                }
            )
            aeid_counter += 1

    return pd.DataFrame(rows)


# ---------------------------
# Validation (v1)
# ---------------------------

def validate_tables(cfg: GenConfig, dm: pd.DataFrame, mh: pd.DataFrame, vs: pd.DataFrame, lb: pd.DataFrame, ae: pd.DataFrame) -> List[str]:
    issues = []

    # DM: USUBJID unique
    if dm["USUBJID"].duplicated().any():
        issues.append("DM: duplicate USUBJID found.")

    subj_set = set(dm["USUBJID"].astype(str).tolist())

    # FK checks
    for name, df in [("MH", mh), ("VS", vs), ("LB", lb), ("AE", ae)]:
        if len(df) == 0:
            continue
        bad = set(df["USUBJID"].astype(str).tolist()) - subj_set
        if bad:
            issues.append(f"{name}: FK violation USUBJID not in DM: {sorted(list(bad))[:5]}...")

    # VS/LB: exactly one row per subject per visit
    expected = cfg.n_subjects * len(cfg.visits)
    if len(vs) != expected:
        issues.append(f"VS: expected {expected} rows, found {len(vs)}.")
    if len(lb) != expected:
        issues.append(f"LB: expected {expected} rows, found {len(lb)}.")

    # Visit integrity: valid visit labels and nums
    valid_visits = set([v.visit for v in cfg.visits])
    valid_nums = set([v.visitnum for v in cfg.visits])

    for name, df in [("VS", vs), ("LB", lb)]:
        if not set(df["VISIT"].unique()).issubset(valid_visits):
            issues.append(f"{name}: unexpected VISIT values.")
        if not set(df["VISITNUM"].unique()).issubset(valid_nums):
            issues.append(f"{name}: unexpected VISITNUM values.")

        # Date ordering per subject
        for usubjid, g in df.groupby("USUBJID"):
            dts = [datetime.fromisoformat(x).date() for x in g.sort_values("VISITNUM")["VISITDT"].tolist()]
            if any(dts[i] > dts[i+1] for i in range(len(dts)-1)):
                issues.append(f"{name}: VISITDT not increasing for subject {usubjid}.")
                break

    # AE date logic: end >= start
    if len(ae) > 0:
        s = pd.to_datetime(ae["AESTDTC"], errors="coerce")
        e = pd.to_datetime(ae["AEENDTC"], errors="coerce")
        if (e < s).any():
            issues.append("AE: AEENDTC earlier than AESTDTC for some rows.")

        # Severe -> AESER must be Y
        bad_severe = ae[(ae["AESEV"] == "SEVERE") & (ae["AESER"] != "Y")]
        if len(bad_severe) > 0:
            issues.append("AE: severe AE with AESER != Y found.")

    return issues


# ---------------------------
# Export
# ---------------------------

def make_zip_bytes(files: Dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for fname, content in files.items():
            zf.writestr(fname, content)
    return buf.getvalue()


def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


# ---------------------------
# Streamlit UI
# ---------------------------

st.set_page_config(page_title="Synthetic EDC Data Generator (v1)", layout="wide")
st.title("Synthetic EDC Data Generator (v1)")
st.caption("v1: No LLM parsing. Uses a fixed demo schema with integrity checks. Output = ZIP of CSVs + manifest.")

with st.sidebar:
    st.header("Controls")
    seed = st.number_input("Seed", min_value=0, max_value=10_000_000, value=42, step=1)
    n_subjects = st.slider("Number of subjects", min_value=10, max_value=500, value=100, step=10)
    n_sites = st.slider("Number of sites", min_value=1, max_value=50, value=5, step=1)
    severe_rate = st.slider("Severe AE rate (among AEs)", min_value=0.0, max_value=0.8, value=0.20, step=0.05)
    ae_mean = st.slider("Mean AEs per subject (Poisson)", min_value=0.0, max_value=3.0, value=0.6, step=0.1)

prompt = st.text_area(
    "Prompt (stored in manifest; ignored by v1 generator)",
    value=(
        "You are a synthetic data generator agent for clinical trial EDC system. "
        "Generate 100 patients across treatment and placebo cohort for rheumatoid arthritis phase 2 trial, "
        "across 5 visit (baseline, 2 weeks, 4, 6, 8) along with demography, medical history (static) and "
        "labs, vitals and adverse events (20% chance of severe AE) longitudinally."
    ),
    height=120,
)

colA, colB = st.columns([1, 1])

if colA.button("Generate dataset", type="primary"):
    cfg = GenConfig(
        n_subjects=int(n_subjects),
        n_sites=int(n_sites),
        severe_ae_rate=float(severe_rate),
        ae_mean_per_subject=float(ae_mean),
    )
    _set_seed(int(seed))
    fake = Faker()
    Faker.seed(int(seed))

    dm = generate_dm(cfg, fake)
    mh = generate_mh(cfg, dm)
    vs = generate_vs(cfg, dm)
    lb = generate_lb(cfg, dm)
    ae = generate_ae(cfg, dm)

    issues = validate_tables(cfg, dm, mh, vs, lb, ae)

    manifest = {
        "version": "v1",
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "seed": int(seed),
        "prompt": prompt,
        "config_used": {
            "studyid": cfg.studyid,
            "disease": cfg.disease,
            "phase": cfg.phase,
            "n_subjects": cfg.n_subjects,
            "n_sites": cfg.n_sites,
            "arms": list(cfg.arms),
            "arm_ratio": list(cfg.arm_ratio),
            "visits": [{"visit": v.visit, "visitnum": v.visitnum} for v in cfg.visits],
            "severe_ae_rate": cfg.severe_ae_rate,
            "ae_mean_per_subject": cfg.ae_mean_per_subject,
        },
        "row_counts": {
            "DM": int(len(dm)),
            "MH": int(len(mh)),
            "VS": int(len(vs)),
            "LB": int(len(lb)),
            "AE": int(len(ae)),
        },
        "validation_issues": issues,
    }

    zip_bytes = make_zip_bytes(
        {
            "DM.csv": df_to_csv_bytes(dm),
            "MH.csv": df_to_csv_bytes(mh),
            "VS.csv": df_to_csv_bytes(vs),
            "LB.csv": df_to_csv_bytes(lb),
            "AE.csv": df_to_csv_bytes(ae),
            "manifest.json": json.dumps(manifest, indent=2).encode("utf-8"),
        }
    )

    st.success("Generated.")
    if issues:
        st.warning("Validation issues found (v1 is basic):")
        for it in issues:
            st.write(f"- {it}")
    else:
        st.info("Validation passed (basic checks).")

    st.download_button(
        "Download ZIP (CSVs + manifest)",
        data=zip_bytes,
        file_name="syn_edc_demo.zip",
        mime="application/zip",
    )

    # Previews
    with st.expander("Preview: DM"):
        st.dataframe(dm.head(50), use_container_width=True)
    with st.expander("Preview: MH"):
        st.dataframe(mh.head(50), use_container_width=True)
    with st.expander("Preview: VS"):
        st.dataframe(vs.head(50), use_container_width=True)
    with st.expander("Preview: LB"):
        st.dataframe(lb.head(50), use_container_width=True)
    with st.expander("Preview: AE"):
        st.dataframe(ae.head(50), use_container_width=True)

colB.markdown(
    """
**What this v1 demonstrates**
- 5-table relational output (PK/FK integrity)
- fixed visit schedule (baseline, wk2, wk4, wk6, wk8)
- longitudinal VS/LB per visit
- AE generation with severe fraction and date logic
- reproducible generation via seed
"""
)
