import re
import sys
import hashlib
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# Configuration
RANDOM_STATE = 42
DATASET_VERSION = "mpea_dataset_processed_v16"
R = 8.314

# Environment versions
PY_VERSION = sys.version.split()[0]
NP_VERSION = np.__version__
PD_VERSION = pd.__version__
SK_VERSION = sklearn.__version__

# Data limits
MIN_ELEMENTS = 2
MAX_ELEMENT_FRACTION = 0.90
ELEMENT_ACTIVITY_THRESHOLD = 0.01

MIN_HV = 50.0
MAX_HV = 1200.0

# Signature precision
SIGNATURE_PRECISION = 3

# Coverage thresholds
HMIX_PAIR_COVERAGE_MIN = 0.95
HMIX_DATASET_TOLERANCE = 0.95

# Files
BORG_PATH = Path("MPEA_dataset.csv")
MIEDEMA_PATH = Path("miedema_enthalpy_template_VERIFIED.csv")

required_files = [BORG_PATH, MIEDEMA_PATH]

if IN_COLAB and any(not path.exists() for path in required_files):
    print(
        "Upload input CSV files "
        "(MPEA_dataset.csv and miedema_enthalpy_template_VERIFIED.csv):"
    )
    files.upload()

missing_files = [str(path) for path in required_files if not path.exists()]
if missing_files:
    raise FileNotFoundError(f"Missing required input files: {missing_files}")

OUT_DIR = Path("output")
OUT_DIR.mkdir(exist_ok=True)

MASTER_OUT = OUT_DIR / "MASTER_MPEA_DATASET.csv"
META_OUT = OUT_DIR / "MASTER_METADATA.txt"
AUDIT_OUT = OUT_DIR / "Dataset_Audit.txt"
PARSE_ERRORS_OUT = OUT_DIR / "PARSE_FAILURES.txt"
ELEMENT_FREQ_OUT = OUT_DIR / "ELEMENT_FREQUENCY.csv"
SAMPLE_ID_OUT = OUT_DIR / "SAMPLE_IDS.csv"
SPLIT_OUT = OUT_DIR / "DATA_SPLITS.csv"
MISSING_ENTHALPY_OUT = OUT_DIR / "MISSING_ENTHALPY_PAIRS.txt"
PHASE_VALIDATION_OUT = OUT_DIR / "PHASE_VALIDATION.csv"

# Supported elements
SUPPORTED_ELEMENTS = {
    "Ag",
    "Al",
    "B",
    "C",
    "Ca",
    "Co",
    "Cr",
    "Cu",
    "Fe",
    "Ga",
    "Hf",
    "Li",
    "Mg",
    "Mn",
    "Mo",
    "Nb",
    "Nd",
    "Ni",
    "Pd",
    "Re",
    "Sc",
    "Si",
    "Sn",
    "Ta",
    "Ti",
    "V",
    "W",
    "Y",
    "Zn",
    "Zr",
}

AUDIT_TRAIL = {}
UNSUPPORTED_TRACKER = set()
PARSE_FAILURES = []
MISSING_ENTHALPY_PAIRS = set()

# Element properties
ELEMENT_PROPS = {
    "Sc": {"VEC": 3.0, "r": 1.62, "Tm": 1814.0, "chi": 1.36},
    "Ti": {"VEC": 4.0, "r": 1.47, "Tm": 1941.0, "chi": 1.54},
    "V": {"VEC": 5.0, "r": 1.34, "Tm": 2183.0, "chi": 1.63},
    "Cr": {"VEC": 6.0, "r": 1.28, "Tm": 2180.0, "chi": 1.66},
    "Mn": {"VEC": 7.0, "r": 1.27, "Tm": 1519.0, "chi": 1.55},
    "Fe": {"VEC": 8.0, "r": 1.26, "Tm": 1811.0, "chi": 1.83},
    "Co": {"VEC": 9.0, "r": 1.25, "Tm": 1768.0, "chi": 1.88},
    "Ni": {"VEC": 10.0, "r": 1.24, "Tm": 1728.0, "chi": 1.91},
    "Cu": {"VEC": 11.0, "r": 1.28, "Tm": 1358.0, "chi": 1.90},
    "Zn": {"VEC": 12.0, "r": 1.34, "Tm": 693.0, "chi": 1.65},
    "Y": {"VEC": 3.0, "r": 1.80, "Tm": 1799.0, "chi": 1.22},
    "Zr": {"VEC": 4.0, "r": 1.60, "Tm": 2128.0, "chi": 1.33},
    "Nb": {"VEC": 5.0, "r": 1.46, "Tm": 2750.0, "chi": 1.60},
    "Mo": {"VEC": 6.0, "r": 1.39, "Tm": 2896.0, "chi": 2.16},
    "Pd": {"VEC": 10.0, "r": 1.37, "Tm": 1828.0, "chi": 2.20},
    "Hf": {"VEC": 4.0, "r": 1.59, "Tm": 2506.0, "chi": 1.30},
    "Ta": {"VEC": 5.0, "r": 1.46, "Tm": 3290.0, "chi": 1.50},
    "W": {"VEC": 6.0, "r": 1.39, "Tm": 3695.0, "chi": 2.36},
    "Al": {"VEC": 3.0, "r": 1.43, "Tm": 933.5, "chi": 1.61},
    "Ga": {"VEC": 3.0, "r": 1.22, "Tm": 302.9, "chi": 1.81},
    "Si": {"VEC": 4.0, "r": 1.17, "Tm": 1687.0, "chi": 1.90},
    "Sn": {"VEC": 4.0, "r": 1.40, "Tm": 505.0, "chi": 1.96},
    "B": {"VEC": 3.0, "r": 0.82, "Tm": 2349.0, "chi": 2.04},
    "C": {"VEC": 4.0, "r": 0.77, "Tm": 3800.0, "chi": 2.55},
    "Ag": {"VEC": 11.0, "r": 1.44, "Tm": 1234.9, "chi": 1.93},
    "Ca": {"VEC": 2.0, "r": 1.97, "Tm": 1115.0, "chi": 1.00},
    "Li": {"VEC": 1.0, "r": 1.52, "Tm": 453.7, "chi": 0.98},
    "Mg": {"VEC": 2.0, "r": 1.60, "Tm": 923.0, "chi": 1.31},
    "Nd": {"VEC": 3.0, "r": 1.82, "Tm": 1297.0, "chi": 1.14},
    "Re": {"VEC": 7.0, "r": 1.37, "Tm": 3459.0, "chi": 1.90},
}

# Enthalpy data
H_MIX_BINARY = {}

if MIEDEMA_PATH.exists():
    miedema_df = pd.read_csv(MIEDEMA_PATH)

    for _, row in miedema_df.iterrows():
        elem1 = str(row["Element1"]).strip()
        elem2 = str(row["Element2"]).strip()
        pair_key = tuple(sorted([elem1, elem2]))
        H_MIX_BINARY[pair_key] = float(row["H_mix"])
else:
    print(f"WARNING: {MIEDEMA_PATH.name} not found.")


def log_audit(step_name, dataframe):
    AUDIT_TRAIL[step_name] = len(dataframe)


def parse_formula(formula):
    if pd.isna(formula):
        return {}

    formula_str = str(formula).strip().replace(" ", "")
    pattern = r"([A-Z][a-z]*)([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)?"
    parts = re.findall(pattern, formula_str)

    composition = {}

    for element, amount in parts:
        if element not in SUPPORTED_ELEMENTS:
            UNSUPPORTED_TRACKER.add(element)
            continue

        if amount in ["", None]:
            amount = 1.0
        else:
            try:
                amount = float(amount)
            except ValueError:
                PARSE_FAILURES.append(f"Malformed amount in {formula_str}")
                return {}

        if amount < 0:
            PARSE_FAILURES.append(f"Negative stoichiometry in {formula_str}")
            return {}

        composition[f"ELEM_{element}"] = amount

    total = sum(composition.values())

    if total <= 0:
        PARSE_FAILURES.append(f"Zero/Negative total in {formula_str}")
        return {}

    return {key: value / total for key, value in composition.items()}


def normalize_process_condition(value):
    if pd.isna(value):
        return "UNKNOWN"

    text = str(value).upper()

    if any(
        key in text
        for key in ["ANNEAL", "HEAT", "SOLUTION", "HOMOGEN"]
    ):
        return "HEAT_TREATED"

    if any(key in text for key in ["ROLL", "FORGE", "WROUGHT"]):
        return "DEFORMED"

    if any(key in text for key in ["SPS", "POWDER", "PM"]):
        return "POWDER_MET"

    if any(key in text for key in ["AS-CAST", "CAST"]):
        return "AS_CAST"

    return "OTHER"


def normalize_observed_phase(micro_val, phase_val):
    if pd.isna(micro_val):
        micro_val = ""

    if pd.isna(phase_val):
        phase_val = ""

    text = f"{micro_val} {phase_val}".upper()

    if not text.strip() or text.strip() == "NAN NAN":
        return "UNKNOWN"

    if "FCC+BCC" in text or (
        "FCC" in text and "BCC" in text and "B2" not in text
    ):
        return "FCC_BCC"

    if "BCC" in text and "B2" in text:
        return "BCC_B2"

    if "FCC" in text:
        return "FCC"

    if "BCC" in text:
        return "BCC"

    if "HCP" in text:
        return "HCP"

    if any(key in text for key in ["AMORPH", "GLASS"]):
        return "AMORPHOUS"

    if any(
        key in text
        for key in ["INTERMETALLIC", "LAVES", "B2", "SIGMA", "MU", "L12"]
    ):
        return "INTERMETALLIC"

    return "UNKNOWN"


def classify_vec_phase_tendency(vec):
    """Used ONLY for validation to capture theoretical phase tendency. Excluded from ML features."""
    if pd.isna(vec) or vec == 0:
        return "UNKNOWN"

    if vec >= 8.0:
        return "FCC"
    elif vec <= 6.87:
        return "BCC"
    else:
        return "FCC_BCC"


def map_phase_group(phase):
    """Maps complex experimental phases to base thermodynamic predictions for validation."""
    if phase == "BCC_B2":
        return "BCC"

    if phase in ["FCC", "BCC", "FCC_BCC"]:
        return phase

    return "OTHER"


def load_borg_dataset(path):
    raw = pd.read_csv(path)
    data = raw.dropna(subset=["PROPERTY: HV"]).copy()
    original_indices = data.index.astype(str)
    data = data.reset_index(drop=True)

    elements = pd.DataFrame(
        data["FORMULA"].apply(parse_formula).tolist()
    ).fillna(0)

    elements["PROPERTY: HV"] = data["PROPERTY: HV"]
    elements["RAW_FORMULA"] = data["FORMULA"]
    elements["FORMULA"] = data["FORMULA"]
    elements["SOURCE_ROW_ID"] = "BORG_" + original_indices

    if "PROPERTY: Processing method" in data.columns:
        elements["PROCESSING"] = data["PROPERTY: Processing method"]
    else:
        elements["PROCESSING"] = np.nan

    temp_col_name = r"PROPERTY: Test temperature ($^\circ$C)"

    if temp_col_name in data.columns:
        temp_series = pd.to_numeric(data[temp_col_name], errors="coerce")
    else:
        temp_series = pd.Series([np.nan] * len(data))

    elements["TEST_TEMP"] = temp_series
    elements["TEST_TEMP_BIN"] = np.round(temp_series.fillna(25) / 25) * 25

    if "PROPERTY: Microstructure" in data.columns:
        micro_col = data["PROPERTY: Microstructure"]
    else:
        micro_col = pd.Series([np.nan] * len(data))

    if "PROPERTY: BCC/FCC/other" in data.columns:
        phase_col = data["PROPERTY: BCC/FCC/other"]
    else:
        phase_col = pd.Series([np.nan] * len(data))

    elements["RAW_PROCESSING"] = elements["PROCESSING"]
    elements["RAW_MICROSTRUCTURE"] = micro_col
    elements["RAW_PHASE"] = phase_col
    elements["PHASE_OBSERVED"] = [
        normalize_observed_phase(m, p)
        for m, p in zip(micro_col, phase_col)
    ]
    elements["PROCESS_CONDITION"] = elements["PROCESSING"].apply(
        normalize_process_condition
    )
    elements["SOURCE"] = "BORG"

    return elements


def calculate_advanced_descriptors(df):
    vec_vals = np.zeros(len(df))
    tm_avg_vals = np.zeros(len(df))
    chi_avg_vals = np.zeros(len(df))
    r_avg_vals = np.zeros(len(df))

    active_elems = [
        e for e in ELEMENT_PROPS.keys() if f"ELEM_{e}" in df.columns
    ]

    for elem in active_elems:
        col = f"ELEM_{elem}"
        props = ELEMENT_PROPS[elem]
        vec_vals += df[col] * props["VEC"]
        tm_avg_vals += df[col] * props["Tm"]
        chi_avg_vals += df[col] * props["chi"]
        r_avg_vals += df[col] * props["r"]

    df["VEC"] = vec_vals
    df["TM_AVG"] = tm_avg_vals

    delta_r_sum = np.zeros(len(df))
    delta_chi_sum = np.zeros(len(df))
    safe_r_avg = np.where(r_avg_vals == 0, 1, r_avg_vals)

    for elem in active_elems:
        col = f"ELEM_{elem}"
        props = ELEMENT_PROPS[elem]
        delta_r_sum += df[col] * (
            1.0 - (props["r"] / safe_r_avg)
        ) ** 2
        delta_chi_sum += df[col] * (
            props["chi"] - chi_avg_vals
        ) ** 2

    df["DELTA_RADIUS"] = 100.0 * np.sqrt(delta_r_sum)
    df["DELTA_CHI"] = np.sqrt(delta_chi_sum)

    n_elems = len(active_elems)
    H_matrix = np.zeros((n_elems, n_elems))
    M_found = np.zeros((n_elems, n_elems))

    for i, e1 in enumerate(active_elems):
        for j, e2 in enumerate(active_elems):
            if i < j:
                pair_key = tuple(sorted([e1, e2]))

                if pair_key in H_MIX_BINARY:
                    val = H_MIX_BINARY[pair_key]
                    M_found[i, j] = 1
                    M_found[j, i] = 1
                else:
                    val = 0.0
                    MISSING_ENTHALPY_PAIRS.add(pair_key)

                H_matrix[i, j] = val
                H_matrix[j, i] = val

    C = df[[f"ELEM_{e}" for e in active_elems]].values
    A = (C > 0).astype(int)

    needed_pairs = (
        np.sum(A, axis=1) * (np.sum(A, axis=1) - 1) / 2
    )
    found_pairs = np.sum(A * (A @ M_found), axis=1) / 2.0

    safe_needed = np.where(needed_pairs == 0, 1, needed_pairs)
    df["HMIX_COVERAGE"] = found_pairs / safe_needed

    hmix_vals = 2.0 * np.sum(C * (C @ H_matrix), axis=1)
    df["H_MIX"] = hmix_vals

    # Omega
    df["OMEGA"] = (
        df["TM_AVG"]
        * df["CONFIG_ENTROPY"]
        / (np.maximum(np.abs(df["H_MIX"]), 1e-9) * 1000)
    )

    ideal_mask = np.abs(df["H_MIX"]) < 0.1
    df.loc[ideal_mask, "OMEGA"] = np.nan
    df["OMEGA_FINITE"] = (~ideal_mask).astype(int)

    return df


# Dataset processing
np.random.seed(RANDOM_STATE)

master = load_borg_dataset(BORG_PATH)
master["DATASET_VERSION"] = DATASET_VERSION
INITIAL_COUNT = len(master)
log_audit("Dataset loaded", master)

# Hardness data
master = master.dropna(subset=["PROPERTY: HV"]).copy()

# Process conditions
valid_conditions = [
    "AS_CAST",
    "HEAT_TREATED",
    "DEFORMED",
    "POWDER_MET",
    "UNKNOWN",
    "OTHER",
]
master = master[master["PROCESS_CONDITION"].isin(valid_conditions)]

# Normalize compositions
elem_cols = [
    col for col in master.columns if col.startswith("ELEM_")
]
master[elem_cols] = master[elem_cols].fillna(0)

row_sum = master[elem_cols].sum(axis=1)
row_sum[row_sum == 0] = 1
master[elem_cols] = master[elem_cols].div(row_sum, axis=0)

# Configurational entropy
master["CONFIG_ENTROPY"] = -R * (
    master[elem_cols]
    .replace(0, np.nan)
    .apply(
        lambda row: np.nansum(row * np.log(row)),
        axis=1,
    )
)

# Composition limits
master["NUM_ELEMENTS"] = (
    master[elem_cols] > ELEMENT_ACTIVITY_THRESHOLD
).sum(axis=1)
master["MAX_ELEMENT"] = master[elem_cols].max(axis=1)

master = master.loc[master["NUM_ELEMENTS"] >= MIN_ELEMENTS]
master = master.loc[
    master["MAX_ELEMENT"] <= MAX_ELEMENT_FRACTION
]
master = master.loc[
    (master["PROPERTY: HV"] >= MIN_HV)
    & (master["PROPERTY: HV"] <= MAX_HV)
]

# Thermodynamic descriptors
master = calculate_advanced_descriptors(master)
log_audit("Thermodynamic descriptors", master)

# Enthalpy coverage
good_coverage_fraction = (
    master["HMIX_COVERAGE"] >= HMIX_PAIR_COVERAGE_MIN
).mean()

if good_coverage_fraction < HMIX_DATASET_TOLERANCE:
    print(
        f"\nWARNING: Only {good_coverage_fraction * 100:.1f}% of alloys "
        f"meet the {HMIX_PAIR_COVERAGE_MIN * 100}% pair coverage threshold."
    )
    print("Dropping H_MIX, OMEGA, and OMEGA_FINITE features.")
    master = master.drop(
        columns=["H_MIX", "OMEGA", "OMEGA_FINITE"],
        errors="ignore",
    )
    enthalpy_status = "Dropped globally"
else:
    print(
        f"\nDataset H_mix coverage acceptable "
        f"({good_coverage_fraction * 100:.1f}% of alloys exceed threshold)."
    )
    print("H_MIX and OMEGA retained.")
    print("OMEGA_FINITE marks ideal solutions.")
    enthalpy_status = (
        "H_MIX & OMEGA Kept (Ideal Solutions Flagged)"
    )

if "HMIX_COVERAGE" in master.columns:
    master = master.drop(columns=["HMIX_COVERAGE"])

log_audit("Thermodynamic filtering", master)

# Phase validation
master["PHASE_VEC_TENDENCY"] = master["VEC"].apply(
    classify_vec_phase_tendency
)
master["PHASE_GROUP_OBSERVED"] = master["PHASE_OBSERVED"].apply(
    map_phase_group
)
master["PHASE_AGREEMENT"] = (
    master["PHASE_GROUP_OBSERVED"]
    == master["PHASE_VEC_TENDENCY"]
).astype(float)

# Encode categories
master = pd.get_dummies(
    master,
    columns=["PHASE_OBSERVED", "PROCESS_CONDITION"],
    dummy_na=False,
    dtype=int,
)
log_audit(
    "Categorical encoding",
    master,
)

# Signatures and aggregation
master["COMPOSITION_SIGNATURE"] = [
    hashlib.md5(
        "|".join(
            f"{v:.{SIGNATURE_PRECISION}f}"
            for v in row.round(SIGNATURE_PRECISION)
        ).encode()
    ).hexdigest()
    for _, row in master[elem_cols].iterrows()
]

obs_phase_cols = [
    c for c in master.columns if c.startswith("PHASE_OBSERVED_")
]
process_cols = [
    c for c in master.columns if c.startswith("PROCESS_CONDITION_")
]


def build_signature(row):
    obs_sig = "".join(str(row[c]) for c in obs_phase_cols)
    process_sig = "".join(str(row[c]) for c in process_cols)
    temp_val = row["TEST_TEMP_BIN"]
    temp_sig = str(int(temp_val)) if pd.notna(temp_val) else "25"

    return (
        f"{row['COMPOSITION_SIGNATURE']}_"
        f"{process_sig}_{obs_sig}_{temp_sig}"
    )


master["PROCESS_PHASE_SIGNATURE"] = master.apply(
    build_signature,
    axis=1,
)

hv_stats = (
    master.groupby("PROCESS_PHASE_SIGNATURE")
    .agg(
        PROPERTY_HV_MEAN=("PROPERTY: HV", "mean"),
        PROPERTY_HV_MIN=("PROPERTY: HV", "min"),
        PROPERTY_HV_MAX=("PROPERTY: HV", "max"),
        PROPERTY_HV_STD=("PROPERTY: HV", "std"),
        PROPERTY_HV_COUNT=("PROPERTY: HV", "count"),
        SOURCE_COUNT=("SOURCE_ROW_ID", "nunique"),
    )
    .reset_index()
)

hv_stats["PROPERTY_HV_STD"] = hv_stats[
    "PROPERTY_HV_STD"
].fillna(0)
hv_stats["PROPERTY_HV_RANGE"] = (
    hv_stats["PROPERTY_HV_MAX"] - hv_stats["PROPERTY_HV_MIN"]
)
hv_stats["HV_CV"] = np.where(
    hv_stats["PROPERTY_HV_MEAN"] > 0,
    hv_stats["PROPERTY_HV_STD"] / hv_stats["PROPERTY_HV_MEAN"],
    0,
)

aggregation_dict = {
    **{col: "mean" for col in elem_cols},
    **{col: "max" for col in obs_phase_cols},
    **{col: "max" for col in process_cols},
    "TEST_TEMP_BIN": "first",
    "PROPERTY: HV": "mean",
    "CONFIG_ENTROPY": "mean",
    "VEC": "mean",
    "TM_AVG": "mean",
    "DELTA_RADIUS": "mean",
    "DELTA_CHI": "mean",
    "PHASE_VEC_TENDENCY": "first",
    "PHASE_GROUP_OBSERVED": "first",
    "PHASE_AGREEMENT": "mean",
    "COMPOSITION_SIGNATURE": "first",
    "FORMULA": "first",
    "RAW_FORMULA": "first",
    "RAW_PROCESSING": lambda x: "|".join(
        sorted(set(x.dropna().astype(str)))
    ),
    "RAW_MICROSTRUCTURE": lambda x: "|".join(
        sorted(set(x.dropna().astype(str)))
    ),
    "RAW_PHASE": lambda x: "|".join(
        sorted(set(x.dropna().astype(str)))
    ),
    "SOURCE": lambda x: "|".join(
        sorted(set(x.astype(str)))
    ),
    "SOURCE_ROW_ID": lambda x: "|".join(
        sorted(set(x.astype(str)))
    ),
    "NUM_ELEMENTS": "mean",
    "MAX_ELEMENT": "mean",
    "DATASET_VERSION": "first",
}

if "H_MIX" in master.columns:
    aggregation_dict["H_MIX"] = "mean"

if "OMEGA" in master.columns:
    aggregation_dict["OMEGA"] = "mean"
    aggregation_dict["OMEGA_FINITE"] = "min"

master_merged = master.groupby(
    "PROCESS_PHASE_SIGNATURE",
    as_index=False,
).agg(aggregation_dict)

master = master_merged.merge(
    hv_stats,
    on="PROCESS_PHASE_SIGNATURE",
    how="left",
)

# Uncertainty limit
master = master[master["HV_CV"] <= 0.50]
master = master.reset_index(drop=True)

# Hardness classes
hv_qcut = pd.qcut(
    master["PROPERTY: HV"],
    q=4,
    duplicates="drop",
)
n_bins = len(hv_qcut.cat.categories)
target_labels = [
    "LOW",
    "MEDIUM",
    "HIGH",
    "ULTRA_HIGH",
][:n_bins]

master["HV_CLASS"] = (
    hv_qcut.cat.rename_categories(target_labels)
    if hasattr(hv_qcut, "cat")
    else hv_qcut
)

master["SAMPLE_ID"] = [
    f"MPEA_{i:05d}" for i in range(len(master))
]
master["ALLOY_HASH"] = [
    hashlib.sha256(sig.encode()).hexdigest()
    for sig in master["PROCESS_PHASE_SIGNATURE"]
]

# Train/test split
sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)
global_props = master["HV_CLASS"].value_counts(normalize=True)

selected_fold = None
min_deviation = float("inf")

for train_idx, test_idx in sgkf.split(
    X=master,
    y=master["HV_CLASS"],
    groups=master["COMPOSITION_SIGNATURE"],
):
    test_props = (
        master.iloc[test_idx]["HV_CLASS"]
        .value_counts(normalize=True)
    )
    deviation = np.sum(np.abs(global_props - test_props))

    if deviation < min_deviation:
        min_deviation = deviation
        selected_fold = (train_idx, test_idx)

train_idx, test_idx = selected_fold

split_col = np.array(["TRAIN"] * len(master))
split_col[test_idx] = "TEST"
master["DATA_SPLIT"] = split_col

element_frequency = pd.DataFrame(
    {
        "Element": elem_cols,
        "Frequency": [
            (master[col] > 0).sum() for col in elem_cols
        ],
    }
).sort_values("Frequency", ascending=False)

element_frequency.to_csv(ELEMENT_FREQ_OUT, index=False)

# Export results
validation_cols = [
    "SAMPLE_ID",
    "FORMULA",
    "PROCESS_PHASE_SIGNATURE",
    "PHASE_GROUP_OBSERVED",
    "PHASE_VEC_TENDENCY",
    "PHASE_AGREEMENT",
]
master[validation_cols].to_csv(
    PHASE_VALIDATION_OUT,
    index=False,
)

master = master.drop(
    columns=[
        "PHASE_VEC_TENDENCY",
        "PHASE_GROUP_OBSERVED",
        "PHASE_AGREEMENT",
    ]
)

ml_features = [
    "SAMPLE_ID",
    "FORMULA",
    "TEST_TEMP_BIN",
    "PROPERTY: HV",
    "HV_CV",
    "PROPERTY_HV_RANGE",
    "SOURCE_COUNT",
    "VEC",
    "DELTA_RADIUS",
]

if "H_MIX" in master.columns:
    ml_features.append("H_MIX")

if "OMEGA" in master.columns:
    ml_features.append("OMEGA")
    ml_features.append("OMEGA_FINITE")

ml_features += obs_phase_cols + process_cols

master[ml_features].to_csv(SAMPLE_ID_OUT, index=False)

master[
    [
        "SAMPLE_ID",
        "PROCESS_PHASE_SIGNATURE",
        "COMPOSITION_SIGNATURE",
        "DATA_SPLIT",
    ]
].to_csv(SPLIT_OUT, index=False)

tmp_path = OUT_DIR / "_tmp_dataset.csv"
master.to_csv(tmp_path, index=False)

with open(tmp_path, "rb") as file:
    sha = hashlib.sha256(file.read()).hexdigest()

tmp_path.unlink()
master.to_csv(MASTER_OUT, index=False)

if MISSING_ENTHALPY_PAIRS:
    with open(MISSING_ENTHALPY_OUT, "w") as file:
        file.write(
            "Pairs missing from "
            "miedema_enthalpy_template_VERIFIED.csv "
            "(defaulted to 0):\n"
        )

        for p in sorted(MISSING_ENTHALPY_PAIRS):
            file.write(f"{p[0]}-{p[1]}\n")

if PARSE_FAILURES:
    with open(PARSE_ERRORS_OUT, "w") as file:
        file.write("Formula parse entries:\n")
        file.write("\n".join(PARSE_FAILURES))

audit_lines = [
    f"{step}: {count} records"
    for step, count in AUDIT_TRAIL.items()
]
AUDIT_OUT.write_text(
    "\n".join(audit_lines),
    encoding="utf-8",
)

RETENTION_PERCENT = (
    (len(master) / INITIAL_COUNT) * 100
    if INITIAL_COUNT > 0
    else 0
)

metadata_lines = [
    "--- DATASET METADATA ---",
    f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
    f"Dataset version: {DATASET_VERSION}",
    f"SHA256 Checksum: {sha}",
    "",
    "--- ENVIRONMENT VERSIONS ---",
    f"Python: {PY_VERSION}",
    f"Pandas: {PD_VERSION}",
    f"NumPy:  {NP_VERSION}",
    f"Scikit-Learn: {SK_VERSION}",
    "",
    "--- AUDIT METRICS ---",
    f"Initial loaded rows: {INITIAL_COUNT}",
    f"Final valid rows: {len(master)}",
    f"Retention Percentage: {RETENTION_PERCENT:.2f}%",
    f"Final columns: {len(master.columns)}",
    f"Missing Miedema Pairs logged: {len(MISSING_ENTHALPY_PAIRS)}",
    f"Enthalpy Feature Status: {enthalpy_status}",
    "",
    "--- HARDNESS INTERVAL ---",
    (
        f"{master['PROPERTY: HV'].min():.2f} to "
        f"{master['PROPERTY: HV'].max():.2f} HV"
    ),
]

META_OUT.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

print(f"\nDataset size: {master.shape}")
print(
    f"Train/Test split: "
    f"{(master['DATA_SPLIT'] == 'TRAIN').sum()}/"
    f"{(master['DATA_SPLIT'] == 'TEST').sum()}"
)
print(f"Data retention: {RETENTION_PERCENT:.2f}%")
print("PHASE_VALIDATION.csv written.")
print("Dataset exported.")

if IN_COLAB:
    import shutil

    zip_path = shutil.make_archive(
        "mpea_step1_outputs",
        "zip",
        OUT_DIR,
    )
    print(f"Output archive written: {zip_path}")
    files.download(zip_path)
