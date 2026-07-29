import pandas as pd
import numpy as np
import os
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# Classification thresholds
REFRACTORY_THRESHOLD = 0.50
LIGHT_THRESHOLD = 0.50
CANTOR_THRESHOLD = 0.60
CU_RICH_THRESHOLD = 0.20


# Classification helpers
def classify_alloy(row, elem_cols):
    """Classify the alloy family."""

    # Ti is included with refractory elements.
    refractory_elems = {'W', 'Mo', 'Nb', 'Ta', 'V', 'Zr', 'Hf', 'Ti'}
    lightweight_elems = {'Al', 'Mg', 'Li', 'Sc'}
    cantor_elems = {'Co', 'Cr', 'Fe', 'Mn', 'Ni'}

    refractory_sum = sum(row.get(f"ELEM_{el}", 0) for el in refractory_elems if f"ELEM_{el}" in elem_cols)
    light_sum = sum(row.get(f"ELEM_{el}", 0) for el in lightweight_elems if f"ELEM_{el}" in elem_cols)
    cantor_sum = sum(row.get(f"ELEM_{el}", 0) for el in cantor_elems if f"ELEM_{el}" in elem_cols)

    if refractory_sum >= REFRACTORY_THRESHOLD:
        return "Refractory MPEA"
    elif light_sum >= LIGHT_THRESHOLD:
        return "Lightweight/Al-Rich"
    elif cantor_sum >= CANTOR_THRESHOLD:
        if row.get("ELEM_Co", 0) < 0.01: return "Co-Free Cantor-Derivative"
        if row.get("ELEM_Ni", 0) < 0.01: return "Ni-Free Cantor-Derivative"
        return "Cantor-Derivative"
    elif row.get("ELEM_Cu", 0) >= CU_RICH_THRESHOLD:
        return "Cu-Rich MPEA"
    else:
        return "Complex-Concentrated (Mixed)"


def classify_process(process_str):
    """Map processing history to process classes."""
    p_str = str(process_str).strip().upper()

    # Powder metallurgy
    if any(k in p_str for k in [
        "POWDER", "SINTER", "HIP", "SPS", "P/M", "PM"
    ]):
        return "POWDER_METALLURGY"

    # Additive manufacturing
    if any(k in p_str for k in [
        "SLM", "LPBF", "EBM", "LENS", "LASER", "DED", "WAAM"
    ]):
        return "ADDITIVE_MANUFACTURING"

    # Thermomechanical processing
    if any(k in p_str for k in [
        "ROLL", "ROLLED", "FORG", "FORGED",
        "EXTRUD", "DRAWN", "SWAGED"
    ]):
        return "THERMOMECHANICAL"

    # Heat treatment
    if any(k in p_str for k in [
        "ANNEAL", "ANNEALED",
        "AGE", "AGED",
        "HEAT",
        "SOLUTION",
        "QUENCH",
        "TEMPER"
    ]):
        return "HEAT_TREATED"

    # Casting and melting
    if (
        p_str == "AS"
        or p_str == "AS_CAST"
        or p_str == "CAST"
        or p_str.startswith("AS_")
        or any(k in p_str for k in [
            "CAST",
            "MELT",
            "VACUUM",
            "ARC",
            "ARC MELT",
            "INDUCTION",
            "SUCTION",
            "LEVITATION"
        ])
    ):
        return "CASTING_AND_MELTING"

    return "OTHER"


# Ranking pipeline
def step_6_stratified_ranking(input_csv, out_dir, top_n_per_signature=10):
    print("Step 6: stratified ranking and validation selection")

    if not os.path.exists(input_csv):
        print(f"ERROR: Could not find {input_csv}")
        return None

    print("[1] Loading data...")
    df = pd.read_csv(input_csv)

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    elem_cols = [c for c in df.columns if c.startswith("ELEM_")]

    print("[2] Classifying alloys and processes...")
    df["ALLOY_CLASS"] = df.apply(lambda r: classify_alloy(r, elem_cols), axis=1)

    # Use existing PROCESS_CLASS if available.
    if "PROCESS_CLASS" in df.columns:
        df["PROCESS_CLASS"] = df["PROCESS_CLASS"].fillna(
            df["PROCESS"].apply(classify_process)
        )
    else:
        df["PROCESS_CLASS"] = df["PROCESS"].apply(classify_process)

    print("[3] Sorting candidates...")
    df_sorted = df.sort_values(
        by=["PARETO_OPTIMAL", "RANK_SCORE", "NOVELTY_COSINE", "PREDICTED_HV", "SAMPLE_ID"],
        ascending=[False, False, False, False, True]
    )

    print("[4] Applying stratification...")
    family_cap = df_sorted.groupby(["PROCESS_PHASE_SIGNATURE", "ALLOY_FAMILY"]).head(3)

    # Preserve phase differences within the same class.
    df_stratified = family_cap.drop_duplicates(
        subset=["PROCESS_PHASE_SIGNATURE", "ALLOY_CLASS", "PREDICTED_PHASE"],
        keep="first"
    )

    df_stratified = df_stratified.groupby("PROCESS_PHASE_SIGNATURE").head(top_n_per_signature)
    df_stratified.to_csv(out_path / "TOP_STRATIFIED_DIVERSE_ALLOYS.csv", index=False)

    print(f"    -> Extracted {len(df_stratified)} candidates.")
    print(f"    -> Preserved {df_stratified['ALLOY_CLASS'].nunique()} alloy classes.\n")

    print("[5] Generating exports...")
    exports = []

    # Pareto export
    pareto_alloys = df_sorted[df_sorted["PARETO_OPTIMAL"] == True]
    pareto_alloys.to_csv(out_path / "TOP_PARETO_ALL.csv", index=False)
    exports.append(f"TOP_PARETO_ALL.csv (n={len(pareto_alloys)})")

    # Novelty threshold
    novelty_threshold = df_sorted["NOVELTY_COSINE"].quantile(0.90)
    highly_novel = df_sorted[df_sorted["NOVELTY_COSINE"] >= novelty_threshold]
    highly_novel.to_csv(out_path / "TOP_NOVEL_DECILE.csv", index=False)
    exports.append(f"TOP_NOVEL_DECILE.csv (Threshold: {novelty_threshold:.3f})")

    # FEM stack
    fem_pool = df_sorted
    fem_stack = fem_pool.drop_duplicates(subset=["ALLOY_CLASS", "PREDICTED_PHASE", "PROCESS_CLASS"]).head(100)
    fem_stack.to_csv(out_path / "FEM_VALIDATION_STACK.csv", index=False)
    exports.append(f"FEM_VALIDATION_STACK.csv (n={len(fem_stack)})")

    # Explainability export
    explain_cols = [
        "FORMULA", "PREDICTED_HV", "RANK_SCORE", "NOVELTY_COSINE",
        "NEAREST_TRAIN_FORMULA", "AD_TIER", "PREDICTED_PHASE", "PROCESS",
        "CONFIG_ENTROPY", "OMEGA", "OMEGA_FINITE", "VEC", "DELTA_RADIUS", "H_MIX"
    ]
    available_explain_cols = [c for c in explain_cols if c in df_sorted.columns]
    df_sorted.head(500)[available_explain_cols].to_csv(out_path / "TOP_EXPLAINABLE.csv", index=False)
    exports.append("TOP_EXPLAINABLE.csv")

    # Traceability export
    trace_cols = [
        "FORMULA", "ALLOY_CLASS", "PREDICTED_HV", "NEAREST_TRAIN_FORMULA",
        "NOVELTY_COSINE", "AD_TIER", "PROCESS_SUPPORT", "PHASE_SUPPORT",
        "UNIQUE_PUBLICATIONS", "RANK_SCORE"
    ]
    available_trace_cols = [c for c in trace_cols if c in df_sorted.columns]
    df_sorted.head(500)[available_trace_cols].to_csv(out_path / "TOP_TRACEABLE.csv", index=False)
    exports.append("TOP_TRACEABLE.csv")

    # Balanced phase export
    phases = df_sorted["PREDICTED_PHASE"].dropna().unique()
    balanced_frames = []
    for phase in phases:
        p_df = df_sorted[df_sorted["PREDICTED_PHASE"] == phase]
        p_diverse = p_df.drop_duplicates(subset=["ALLOY_CLASS", "ALLOY_FAMILY"])
        p_diverse_capped = p_diverse.head(min(10, len(p_diverse)))
        balanced_frames.append(p_diverse_capped)

    if balanced_frames:
        balanced_phase_export = pd.concat(balanced_frames)
        balanced_phase_export.to_csv(out_path / "BALANCED_PHASE_EXPORT.csv", index=False)
        exports.append("BALANCED_PHASE_EXPORT.csv")

    # Process class export
    proc_unique = df_sorted.drop_duplicates(subset=["PROCESS_CLASS", "ALLOY_CLASS"], keep="first")
    proc_unique.groupby("PROCESS_CLASS").head(10).to_csv(out_path / "TOP_PROCESS_CLASS.csv", index=False)
    exports.append("TOP_PROCESS_CLASS.csv")

    print("\n    -> Compiled datasets:")
    for ex in sorted(exports):
        print(f"        {ex}")

    print("\nPipeline complete: validation datasets generated")

    return df_stratified, out_path


# Execution
INPUT_FILE = "output/step5_unified_generation/GENERATED_ALLOYS_RANKED.csv"
OUT_DIR = "output/step6_validation_exports"

if __name__ == "__main__":
    result = step_6_stratified_ranking(INPUT_FILE, OUT_DIR, top_n_per_signature=10)

    if result is not None and IN_COLAB:
        df_stratified, export_path = result
        print("\nTriggering browser downloads for key files...")

        # Main stratified list
        files.download(export_path / "TOP_STRATIFIED_DIVERSE_ALLOYS.csv")

        # Explainability export
        if (export_path / "TOP_EXPLAINABLE.csv").exists():
            files.download(export_path / "TOP_EXPLAINABLE.csv")

        # FEM validation stack
        if (export_path / "FEM_VALIDATION_STACK.csv").exists():
            files.download(export_path / "FEM_VALIDATION_STACK.csv")
