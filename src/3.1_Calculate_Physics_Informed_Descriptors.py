import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


# Configuration
DB_VERSION = "step3a_property_db_v10"
REFERENCE_STATE = "Room-temperature equilibrium bulk elemental properties"
PROPERTY_USAGE_NOTE = (
    "Scalar elemental properties are used as transferable composition-based "
    "descriptor inputs. They do not explicitly model temperature-dependent "
    "phase transitions or polymorphic state evolution."
)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Output files
out_dir = Path("output")
out_dir.mkdir(parents=True, exist_ok=True)

MASTER_DATASET_PATH = out_dir / "MASTER_MPEA_DATASET.csv"
out_csv = out_dir / "UNIVERSAL_PROPERTY_DB.csv"
property_group_out = out_dir / "PROPERTY_GROUPS.txt"
units_out = out_dir / "PROPERTY_UNITS.json"
taxonomy_out = out_dir / "PROPERTY_TAXONOMY.json"
metadata_out = out_dir / "PROPERTY_DB_METADATA.txt"
descriptor_family_out = out_dir / "DESCRIPTOR_FAMILY_TABLE.csv"
property_data_context_out = out_dir / "PROPERTY_DATA_CONTEXT.json"

# Load data
if IN_COLAB and not MASTER_DATASET_PATH.exists():
    print("Upload MASTER_MPEA_DATASET.csv (from Step 1)")
    files.upload()

    uploaded_master = Path("MASTER_MPEA_DATASET.csv")
    if uploaded_master.exists():
        shutil.copyfile(uploaded_master, MASTER_DATASET_PATH)

if not MASTER_DATASET_PATH.exists():
    raise FileNotFoundError(
        f"Step 1 output not found at {MASTER_DATASET_PATH}."
    )

PROPERTY_UNITS = {
    "atomic_mass": "g/mol",
    "r": "Angstrom",
    "atomic_volume": "cm3/mol",
    "rho": "g/cm3",
    "Tm": "K",
    "E": "GPa",
    "G": "GPa",
    "K": "GPa",
    "thermal_cond": "W/mK",
    "cohesive_energy": "eV/atom",
    "work_function": "eV",
    "chi": "Pauling",
    "poisson_ratio": "dimensionless",
    "Pugh_ratio": "dimensionless",
    "reduced_modulus_proxy": "GPa",
    "specific_stiffness": "GPa.cm3/g",
    "stiffness_density_index": "GPa.cm3/g",
    "Tm_over_rho": "K.cm3/g",
    "thermal_cond_specific": "W.cm3/gK",
    "modulus_to_melting_ratio": "GPa/K",
}

PROPERTY_TAXONOMY = {
    "canonical_descriptors": [
        "atomic_mass",
        "r",
        "atomic_volume",
        "rho",
        "Tm",
        "E",
        "G",
        "K",
        "VEC",
        "thermal_cond",
        "cohesive_energy",
        "work_function",
        "chi",
    ],
    "derived_descriptors": [
        "poisson_ratio",
        "Pugh_ratio",
        "reduced_modulus_proxy",
        "specific_stiffness",
        "stiffness_density_index",
        "Tm_over_rho",
        "thermal_cond_specific",
        "modulus_to_melting_ratio",
    ],
    "metadata_only": [
        "group",
        "period",
        "crystal",
        "role",
        "d_block",
        "f_block",
        "polymorphic_element",
        "is_refractory",
        "is_fcc_stabilizer",
        "is_bcc_stabilizer",
        "is_lightweight",
        "is_rare_earth",
        "is_metalloid",
    ],
}

PROPERTY_DATA_CONTEXT = {
    "canonical": (
        "ASM Handbook, CRC Handbook, and standard metallurgy references"
    ),
    "cohesive_energy": "Materials thermodynamics literature",
    "work_function": "Surface science literature",
    "chi": "Pauling electronegativity scale",
    "derived_physical": (
        "Relations derived from canonical elemental properties"
    ),
}

DESCRIPTOR_FAMILY_MAP = {
    "atomic_mass": "mass",
    "r": "atomic_size",
    "atomic_volume": "atomic_size",
    "rho": "density",
    "Tm": "thermodynamic",
    "E": "elastic",
    "G": "elastic",
    "K": "elastic",
    "VEC": "electronic",
    "thermal_cond": "transport",
    "cohesive_energy": "bonding",
    "work_function": "electronic",
    "chi": "electronic",
    "poisson_ratio": "elastic_ratio",
    "Pugh_ratio": "elastic_ratio",
    "reduced_modulus_proxy": "elastic",
    "specific_stiffness": "density_normalized",
    "stiffness_density_index": "density_normalized",
    "Tm_over_rho": "density_normalized",
    "thermal_cond_specific": "transport",
    "modulus_to_melting_ratio": "thermoelastic",
}

# Element properties
data = [
    # Element, atomic_mass, r, atomic_volume, rho, Tm, E, G, K, VEC, thermal_cond, cohesive_energy, work_function, chi, group, period, crystal, role
    ["Ag", 107.87, 1.44, 10.3, 10.49, 1234.9, 83, 30, 100, 11.0, 429, 2.95, 4.26, 1.93, 11, 5, "FCC", "fcc_stabilizer"],
    ["Al", 26.98, 1.43, 10.0, 2.70, 933.5, 70, 26, 76, 3.0, 237, 3.39, 4.2, 1.61, 13, 3, "FCC", "lightweight"],
    ["B", 10.81, 0.82, 4.6, 2.34, 2349.0, 400, 160, 320, 3.0, 27.4, 5.81, 4.45, 2.04, 13, 2, "RHOMBOHEDRAL", "metalloid"],
    ["C", 12.01, 0.77, 3.4, 3.51, 3800.0, 1050, 478, 442, 4.0, 2000, 7.37, 5.0, 2.55, 14, 2, "DIAMOND", "metalloid"],
    ["Ca", 40.08, 1.97, 26.2, 1.55, 1115.0, 20, 7.4, 17, 2.0, 201, 1.84, 2.90, 1.00, 2, 4, "FCC", "lightweight"],
    ["Co", 58.93, 1.25, 6.7, 8.90, 1768.0, 211, 75, 180, 9.0, 100, 4.39, 5.0, 1.88, 9, 4, "HCP", "fcc_stabilizer"],
    ["Cr", 52.00, 1.28, 7.2, 7.19, 2180.0, 279, 115, 160, 6.0, 94, 4.10, 4.5, 1.66, 6, 4, "BCC", "bcc_stabilizer"],
    ["Cu", 63.55, 1.28, 7.1, 8.96, 1358.0, 130, 48, 140, 11.0, 401, 3.49, 4.7, 1.90, 11, 4, "FCC", "fcc_stabilizer"],
    ["Fe", 55.85, 1.26, 7.1, 7.87, 1811.0, 211, 82, 170, 8.0, 80, 4.28, 4.5, 1.83, 8, 4, "BCC", "transition"],
    ["Ga", 69.72, 1.22, 11.8, 5.91, 302.9, 10, 3.3, 15, 3.0, 40.6, 2.81, 4.32, 1.81, 13, 4, "ORTHORHOMBIC", "transition"],
    ["Ge", 72.63, 1.22, 13.6, 5.32, 1211.0, 103, 41, 75, 4.0, 60.2, 3.85, 5.0, 2.01, 14, 4, "DIAMOND", "metalloid"],
    ["Hf", 178.49, 1.59, 13.6, 13.31, 2506.0, 78, 30, 110, 4.0, 23, 6.44, 3.9, 1.30, 4, 6, "HCP", "refractory"],
    ["Li", 6.94, 1.52, 13.1, 0.53, 453.7, 4.9, 4.2, 11, 1.0, 85, 1.63, 2.93, 0.98, 1, 2, "BCC", "lightweight"],
    ["Mg", 24.31, 1.60, 14.0, 1.74, 923.0, 45, 17, 35, 2.0, 156, 1.51, 3.66, 1.31, 2, 3, "HCP", "lightweight"],
    ["Mn", 54.94, 1.27, 7.4, 7.21, 1519.0, 198, 76, 120, 7.0, 7.8, 2.92, 4.1, 1.55, 7, 4, "COMPLEX", "transition"],
    ["Mo", 95.95, 1.39, 9.4, 10.28, 2896.0, 329, 126, 230, 6.0, 138, 6.82, 4.6, 2.16, 6, 5, "BCC", "refractory"],
    ["Nb", 92.91, 1.46, 10.8, 8.57, 2750.0, 105, 38, 170, 5.0, 54, 7.57, 4.3, 1.60, 5, 5, "BCC", "refractory"],
    ["Nd", 144.24, 1.82, 20.6, 7.01, 1297.0, 41, 16, 32, 3.0, 17, 3.14, 3.20, 1.14, 3, 6, "HCP", "rare_earth"],
    ["Ni", 58.69, 1.24, 6.6, 8.90, 1728.0, 200, 76, 180, 10.0, 91, 4.44, 5.2, 1.91, 10, 4, "FCC", "fcc_stabilizer"],
    ["Pd", 106.42, 1.37, 8.9, 12.02, 1828.0, 121, 44, 180, 10.0, 72, 3.89, 5.1, 2.20, 10, 5, "FCC", "fcc_stabilizer"],
    ["Pt", 195.08, 1.39, 9.1, 21.45, 2041.0, 168, 61, 230, 10.0, 71.6, 5.84, 5.65, 2.28, 10, 6, "FCC", "transition"],
    ["Re", 186.21, 1.37, 8.9, 21.02, 3459.0, 463, 178, 370, 7.0, 48, 8.03, 4.96, 1.90, 7, 6, "HCP", "refractory"],
    ["Rh", 102.91, 1.34, 8.3, 12.41, 2237.0, 380, 150, 275, 9.0, 150, 5.75, 4.98, 2.28, 9, 5, "FCC", "transition"],
    ["Ru", 101.07, 1.34, 8.3, 12.37, 2607.0, 447, 173, 320, 8.0, 117, 6.74, 4.7, 2.20, 8, 5, "HCP", "transition"],
    ["Sc", 44.96, 1.62, 15.0, 2.99, 1814.0, 74, 29, 57, 3.0, 16, 3.90, 3.5, 1.36, 3, 4, "HCP", "rare_earth"],
    ["Si", 28.09, 1.17, 12.1, 2.33, 1687.0, 130, 51, 98, 4.0, 149, 4.63, 4.8, 1.90, 14, 3, "DIAMOND", "metalloid"],
    ["Sn", 118.71, 1.40, 16.3, 7.31, 505.0, 50, 18, 58, 4.0, 66.8, 3.14, 4.42, 1.96, 14, 5, "TETRAGONAL", "transition"],
    ["Ta", 180.95, 1.46, 10.9, 16.69, 3290.0, 186, 69, 200, 5.0, 57, 8.10, 4.2, 1.50, 5, 6, "BCC", "refractory"],
    ["Ti", 47.87, 1.47, 10.6, 4.50, 1941.0, 116, 44, 110, 4.0, 22, 4.85, 4.3, 1.54, 4, 4, "HCP", "transition"],
    ["V", 50.94, 1.34, 8.4, 6.11, 2183.0, 128, 47, 160, 5.0, 31, 5.31, 4.3, 1.63, 5, 4, "BCC", "bcc_stabilizer"],
    ["W", 183.84, 1.39, 9.5, 19.25, 3695.0, 411, 161, 310, 6.0, 173, 8.90, 4.5, 2.36, 6, 6, "BCC", "refractory"],
    ["Y", 88.91, 1.80, 19.9, 4.47, 1799.0, 64, 26, 41, 3.0, 17, 4.37, 3.1, 1.22, 3, 5, "HCP", "rare_earth"],
    ["Zn", 65.38, 1.34, 9.2, 7.14, 693.0, 108, 43, 70, 12.0, 116, 1.35, 4.3, 1.65, 12, 4, "HCP", "transition"],
    ["Zr", 91.22, 1.60, 14.0, 6.52, 2128.0, 88, 33, 92, 4.0, 23, 6.25, 4.1, 1.33, 4, 5, "HCP", "refractory"],
]

columns = [
    "Element",
    "atomic_mass",
    "r",
    "atomic_volume",
    "rho",
    "Tm",
    "E",
    "G",
    "K",
    "VEC",
    "thermal_cond",
    "cohesive_energy",
    "work_function",
    "chi",
    "group",
    "period",
    "crystal",
    "role",
]

db = pd.DataFrame(data, columns=columns)

numeric_cols = [
    column
    for column in db.columns
    if column not in ["Element", "crystal", "role"]
]
db[numeric_cols] = db[numeric_cols].apply(
    pd.to_numeric,
    errors="coerce",
)

db["property_reference_state"] = REFERENCE_STATE
db["property_usage_note"] = PROPERTY_USAGE_NOTE
db["property_source"] = (
    "Compiled from ASM Handbook, CRC Handbook and peer-reviewed "
    "metallurgy references"
)

# Derived properties
db["poisson_ratio"] = (
    (3 * db["K"] - 2 * db["G"])
    / (2 * (3 * db["K"] + db["G"]) + 1e-9)
)
db["Pugh_ratio"] = db["G"] / (db["K"] + 1e-9)
db["reduced_modulus_proxy"] = (
    db["E"] / (1 - db["poisson_ratio"] ** 2 + 1e-9)
)
db["specific_stiffness"] = db["E"] / (db["rho"] + 1e-9)
db["stiffness_density_index"] = (
    db["G"] / (db["rho"] + 1e-9)
)
db["Tm_over_rho"] = db["Tm"] / (db["rho"] + 1e-9)
db["thermal_cond_specific"] = (
    db["thermal_cond"] / (db["rho"] + 1e-9)
)
db["modulus_to_melting_ratio"] = (
    db["E"] / (db["Tm"] + 1e-9)
)

# Element classifications
db["d_block"] = (
    (db["group"] >= 3) & (db["group"] <= 12)
).astype(int)
db["f_block"] = (
    db["role"] == "rare_earth"
).astype(int)

db["polymorphic_element"] = db["Element"].isin(
    ["Fe", "Ti", "Co", "Zr", "Hf"]
).astype(int)

db["is_refractory"] = (
    db["role"] == "refractory"
).astype(int)
db["is_fcc_stabilizer"] = (
    db["role"] == "fcc_stabilizer"
).astype(int)
db["is_bcc_stabilizer"] = (
    db["role"] == "bcc_stabilizer"
).astype(int)
db["is_lightweight"] = (
    db["role"] == "lightweight"
).astype(int)
db["is_rare_earth"] = (
    db["role"] == "rare_earth"
).astype(int)
db["is_metalloid"] = (
    db["role"] == "metalloid"
).astype(int)

db["DB_VERSION"] = DB_VERSION

# Element coverage
master = pd.read_csv(MASTER_DATASET_PATH)

EXPECTED_ELEMENTS = {
    col.replace("ELEM_", "")
    for col in master.columns
    if col.startswith("ELEM_") and (master[col] > 0).any()
}

db_elements = set(db["Element"])
missing = EXPECTED_ELEMENTS - db_elements

if missing:
    raise ValueError(
        "Element coverage check failed.\n"
        f"Missing properties for active dataset elements: {sorted(missing)}"
    )

# Validate database
if db.isnull().values.any():
    raise ValueError("NaN detected in property database.")

numeric_check = db.select_dtypes(include=np.number)
if np.isinf(numeric_check.values).any():
    raise ValueError("Infinite value detected in property database.")

if db.duplicated("Element").any():
    raise ValueError("Duplicate element detected in property database.")

if not (
    (db["poisson_ratio"] > -1).all()
    and (db["poisson_ratio"] < 0.5).all()
):
    raise ValueError("Poisson ratio outside validation interval.")

if (
    (db["E"] <= 0).any()
    or (db["G"] <= 0).any()
    or (db["K"] <= 0).any()
):
    raise ValueError(
        "Elastic modulus outside positive validation interval."
    )

# Save database
db.to_csv(out_csv, index=False)

with open(units_out, "w", encoding="utf-8") as file:
    json.dump(PROPERTY_UNITS, file, indent=4)

with open(taxonomy_out, "w", encoding="utf-8") as file:
    json.dump(PROPERTY_TAXONOMY, file, indent=4)

with open(
    property_data_context_out,
    "w",
    encoding="utf-8",
) as file:
    json.dump(PROPERTY_DATA_CONTEXT, file, indent=4)

descriptor_family_df = pd.DataFrame(
    {
        "Feature": list(DESCRIPTOR_FAMILY_MAP.keys()),
        "Descriptor_Family": list(DESCRIPTOR_FAMILY_MAP.values()),
    }
)
descriptor_family_df.to_csv(
    descriptor_family_out,
    index=False,
)

property_groups = {
    "Size_and_Mass": [
        "atomic_mass",
        "r",
        "atomic_volume",
        "rho",
    ],
    "Thermodynamic": [
        "Tm",
        "Tm_over_rho",
        "modulus_to_melting_ratio",
    ],
    "Elastic": [
        "E",
        "G",
        "K",
        "poisson_ratio",
        "Pugh_ratio",
        "reduced_modulus_proxy",
        "specific_stiffness",
        "stiffness_density_index",
    ],
    "Electronic": [
        "VEC",
        "work_function",
        "chi",
    ],
    "Bonding": [
        "cohesive_energy",
    ],
    "Transport": [
        "thermal_cond",
        "thermal_cond_specific",
    ],
}

with open(property_group_out, "w", encoding="utf-8") as file:
    for group_name, features in property_groups.items():
        file.write(f"{group_name}\n")

        for feature in features:
            file.write(f"{feature}\n")

        file.write("\n")

# Metadata
db_sha256 = hashlib.sha256(out_csv.read_bytes()).hexdigest()

metadata_lines = [
    "Property database metadata",
    "",
    f"Database version: {DB_VERSION}",
    f"Reference state: {REFERENCE_STATE}",
    f"Usage note: {PROPERTY_USAGE_NOTE}",
    f"Element count: {len(db)}",
    (
        "Canonical descriptor count: "
        f"{len(PROPERTY_TAXONOMY['canonical_descriptors'])}"
    ),
    (
        "Derived descriptor count: "
        f"{len(PROPERTY_TAXONOMY['derived_descriptors'])}"
    ),
    f"SHA256: {db_sha256}",
    "",
    f"Dataset Active Elements: {len(EXPECTED_ELEMENTS)}",
    f"Database Elements: {len(db)}",
    "Coverage Status: COMPLETE",
    "",
]

metadata_out.write_text(
    "\n".join(metadata_lines),
    encoding="utf-8",
)

print(f"Database version: {DB_VERSION}")
print(f"Element count: {len(db)}")
print(f"Dataset Active Elements: {sorted(EXPECTED_ELEMENTS)}")
print(
    "Canonical descriptor count: "
    f"{len(PROPERTY_TAXONOMY['canonical_descriptors'])}"
)
print(
    "Derived descriptor count: "
    f"{len(PROPERTY_TAXONOMY['derived_descriptors'])}"
)
print(f"Output: {out_csv}")
print(f"SHA256: {db_sha256}")
print("Property database written.")

if IN_COLAB:
    zip_path = shutil.make_archive(
        "step3a_property_database_outputs",
        "zip",
        out_dir,
    )
    print(f"Output archive written: {zip_path}")
    files.download(zip_path)
