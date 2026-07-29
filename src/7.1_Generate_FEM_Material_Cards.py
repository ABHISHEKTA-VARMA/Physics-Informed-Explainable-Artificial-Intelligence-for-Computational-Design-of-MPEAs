import hashlib
import json
import shutil
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Configuration
EPS = 1e-12
N_STRAIN_POINTS = 100
ELASTIC_LIMIT_STRAIN = 0.002
MAX_PLASTIC_STRAIN = 0.30  # Indentation profile limit

HARDNESS_TO_YIELD_FACTOR = 3.27  # Approx (9.80665 / 3.0)
HARDENING_FRACTION = 0.10
HARDENING_EXPONENT = 10.0

# Paths
input_path = Path("/content/output/step6_validation_exports/FEM_VALIDATION_STACK.csv")
save_dir = Path("/content/output/step6a_fem_material_cards")
save_dir.mkdir(parents=True, exist_ok=True)

# Element properties: (E_GPa, Density_g_cm3, Poisson)
PURE_PROPS = {
    'Al': (69.0, 2.70, 0.33), 'Co': (209.0, 8.90, 0.31),
    'Cr': (279.0, 7.19, 0.21), 'Cu': (130.0, 8.96, 0.34),
    'Fe': (211.0, 7.87, 0.29), 'Hf': (78.0, 13.31, 0.37),
    'Mg': (45.0, 1.74, 0.29), 'Mn': (198.0, 7.21, 0.28),
    'Mo': (329.0, 10.28, 0.31), 'Nb': (105.0, 8.57, 0.40),
    'Ni': (200.0, 8.90, 0.31), 'Ta': (186.0, 16.69, 0.34),
    'Ti': (116.0, 4.50, 0.32), 'V':  (128.0, 6.11, 0.37),
    'W':  (411.0, 19.25, 0.28), 'Zr': (68.0, 6.52, 0.34),
    'B':  (400.0, 2.34, 0.15), 'C':  (800.0, 3.51, 0.10),
    'Li': (4.9, 0.53, 0.36),  'Nd': (41.4, 7.01, 0.28),
    'Pd': (121.0, 12.02, 0.39), 'Re': (463.0, 21.02, 0.30),
    'Sc': (74.0, 2.98, 0.28), 'Si': (113.0, 2.33, 0.28),
    'Sn': (50.0, 7.31, 0.36), 'Y':  (25.6, 4.47, 0.24),
    'Zn': (108.0, 7.14, 0.25)
}


def compute_rule_of_mixtures(row, elem_cols):
    """Calculate E, density, and Poisson ratio."""
    comp_dict = {c.replace('ELEM_', ''): float(row[c]) for c in elem_cols if float(row[c]) > 0}
    total_frac = sum(comp_dict.values())

    e_gpa, rho_g_cm3, nu = 0.0, 0.0, 0.0

    for el, frac in comp_dict.items():
        norm_frac = frac / (total_frac + EPS)
        el_E, el_rho, el_nu = PURE_PROPS.get(el, (200.0, 7.5, 0.3))  # Fallback

        e_gpa += norm_frac * el_E
        rho_g_cm3 += norm_frac * el_rho
        nu += norm_frac * el_nu

    return e_gpa, rho_g_cm3, nu


def generate_curves(ys_mpa, max_flow_mpa):
    """Generate MISO plasticity curves."""
    ys_pa = ys_mpa * 1e6
    max_flow_pa = max_flow_mpa * 1e6

    yield_strain = ELASTIC_LIMIT_STRAIN
    plastic_strains = np.linspace(0, MAX_PLASTIC_STRAIN, N_STRAIN_POINTS)

    # True plastic curve starts at ys_pa when plastic_strain == 0.
    true_stresses = ys_pa + (max_flow_pa - ys_pa) * (
        plastic_strains / (MAX_PLASTIC_STRAIN + EPS)
    ) ** (1.0 / HARDENING_EXPONENT)

    true_strains = yield_strain + plastic_strains

    # Plastic portion for ANSYS MISO.
    ansys_miso = pd.DataFrame({
        "Plastic_Strain": plastic_strains,
        "True_Stress_Pa": true_stresses
    })

    # Elastic portion for CSV export.
    elastic_df = pd.DataFrame({
        "True_Strain": np.linspace(0, yield_strain, 8),
        "True_Stress_Pa": np.linspace(0, ys_pa, 8),
        "Plastic_Strain": np.zeros(8),
    })

    # Full true stress-strain data.
    true_df = pd.DataFrame({
        "True_Strain": true_strains,
        "True_Stress_Pa": true_stresses,
        "Plastic_Strain": plastic_strains,
    })

    full_true_df = pd.concat([elastic_df, true_df.iloc[1:]], axis=0).reset_index(drop=True)

    eng_df = pd.DataFrame({
        "Engineering_Strain": np.exp(full_true_df["True_Strain"]) - 1,
        "Engineering_Stress_Pa": full_true_df["True_Stress_Pa"] / np.exp(full_true_df["True_Strain"]),
    })

    return full_true_df, eng_df, ansys_miso


# Main pipeline
if not input_path.exists():
    # Fallback to local path.
    input_path = Path("FEM_VALIDATION_STACK.csv")
    if not input_path.exists():
        raise FileNotFoundError(f"Missing representative FEM alloys file: {input_path}")

df = pd.read_csv(input_path)
print(f"Alloys loaded: {len(df)}")

elem_columns = [c for c in df.columns if c.startswith('ELEM_')]
master_summary = []

for idx, row in df.iterrows():
    # Standardize ID.
    raw_id = str(row.get('SAMPLE_ID', idx))
    if "GEN_" in raw_id:
        fem_tag = str(int(raw_id.replace("GEN_", "")))
    else:
        fem_tag = raw_id

    formula = row.get('FORMULA', 'Unknown')
    family = row.get('ALLOY_FAMILY', row.get('ALLOY_CLASS', 'Unknown'))
    hv = float(row.get('PREDICTED_HV', 0.0))

    # Rule of mixtures.
    e_gpa, rho_g_cm3, poisson = compute_rule_of_mixtures(row, elem_columns)
    density_kg_m3 = rho_g_cm3 * 1000.0

    # Plasticity bounds.
    ys_mpa = hv * HARDNESS_TO_YIELD_FACTOR
    max_flow_mpa = ys_mpa * (1 + HARDENING_FRACTION)

    alloy_dir = save_dir / fem_tag
    alloy_dir.mkdir(parents=True, exist_ok=True)

    # Generate curves.
    true_curve, eng_curve, ansys_miso = generate_curves(ys_mpa, max_flow_mpa)

    # Save data.
    true_curve.to_csv(alloy_dir / "ansys_true_stress_strain.csv", index=False)
    eng_curve.to_csv(alloy_dir / "ansys_engineering_stress_strain.csv", index=False)
    ansys_miso.to_csv(alloy_dir / "ansys_multilinear_plasticity.csv", index=False)

    txt_path = alloy_dir / f"{fem_tag}_material_card.txt"
    e_pa = e_gpa * 1e9

    with open(txt_path, "w", encoding="utf-8") as f:
        # Summary
        f.write("ANSYS Material Property Card (Indentation FEA)\n")
        f.write("=========================================\n\n")

        f.write("ANSYS unit system\n")
        f.write("-----------------------------------------\n")
        f.write("Length          : m\n")
        f.write("Force           : N\n")
        f.write("Stress          : Pa\n")
        f.write("Density         : kg/m^3\n")
        f.write("Elastic modulus : Pa\n\n")

        f.write("Alloy identity\n")
        f.write("-----------------------------------------\n")
        f.write(f"FEM Tag     : {fem_tag}\n")
        f.write(f"Class       : {family}\n")
        f.write(f"Composition : {formula}\n\n")

        f.write("Elastic properties\n")
        f.write("-----------------------------------------\n")
        f.write(f"Density (kg/m^3)      : {density_kg_m3:.2f}\n")
        f.write(f"Elastic Modulus (GPa) : {e_gpa:.2f}\n")
        f.write(f"Poisson Ratio         : {poisson:.4f}\n\n")

        f.write("Tensile and plasticity parameters\n")
        f.write("-----------------------------------------\n")
        f.write(f"Vickers Hardness (HV)        : {hv:.2f}\n")
        f.write(f"Yield Strength (MPa)         : {ys_mpa:.2f}\n")
        f.write(f"Maximum Flow Stress (MPa)    : {max_flow_mpa:.2f}\n")
        f.write("Files Generated              : ansys_multilinear_plasticity.csv\n\n")

        # ANSYS APDL macro
        f.write("=========================================\n")
        f.write("ANSYS APDL MACRO (COPY & PASTE)\n")
        f.write("=========================================\n")
        f.write("! --- ELASTIC PROPERTIES (MPDATA) ---\n")
        f.write(f"MP, DENS, 1, {density_kg_m3:.2f}\n")
        f.write(f"MP, EX, 1, {e_pa:.3e}\n")
        f.write(f"MP, PRXY, 1, {poisson:.4f}\n\n")

        f.write("! --- PLASTIC INITIALIZATION (MISO) ---\n")
        f.write(f"TB, MISO, 1, 1, {len(ansys_miso)}\n")
        f.write("TBTEMP, 22.0\n")

        for _, miso_row in ansys_miso.iterrows():
            f.write(f"TBPT, , {miso_row['Plastic_Strain']:.4f}, {miso_row['True_Stress_Pa']:.3e}\n")

    master_summary.append({
        "FEM_Tag": fem_tag,
        "Class": family,
        "Composition": formula,
        "Predicted_HV": hv,
        "Density_kg_m3": density_kg_m3,
        "E_GPa": e_gpa,
        "Poisson_Ratio": poisson,
        "Yield_Strength_MPa": ys_mpa,
        "Max_Flow_Stress_MPa": max_flow_mpa,
    })

master_df = pd.DataFrame(master_summary)
summary_file = save_dir / "master_fem_material_cards.csv"
master_df.to_csv(summary_file, index=False)

print("\nMaterial cards exported to output/step6a_fem_material_cards")

try:
    zip_path = save_dir.parent / "step6a_fem_material_cards"
    shutil.make_archive(base_name=str(zip_path), format="zip", root_dir=save_dir)
    print(f"Archive written: {zip_path}.zip")

    from google.colab import files
    files.download(f"{zip_path}.zip")
except ImportError:
    pass
