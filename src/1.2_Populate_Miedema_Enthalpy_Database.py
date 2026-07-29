!pip install matminer pymatgen pandas
import glob
import os
import sys
from itertools import combinations

import pandas as pd

try:
    from matminer.utils.data import MixingEnthalpy
except ImportError:
    sys.exit("Missing dependency: matminer")

try:
    from pymatgen.core import Element
except ImportError:
    sys.exit("Missing dependency: pymatgen")


# Elements used if no template CSV is available
ELEMENTS = [
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
]

CITATION = (
    "Takeuchi, A.; Inoue, A. Mater. Trans. 2005, 46, 2817-2829. "
    "doi:10.2320/matertrans.46.2817 (table value via matminer "
    "matminer.utils.data.MixingEnthalpy)"
)

NOT_FOUND_NOTE = (
    "NOT in Takeuchi-Inoue (2005) table via matminer -- needs a separate "
    "source"
)


def find_input_csv():
    candidates = sorted(
        glob.glob("miedema_enthalpy_template*.csv"),
        key=os.path.getmtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def load_pairs():
    input_csv = find_input_csv()

    if input_csv:
        df = pd.read_csv(input_csv)
        required = {"Element1", "Element2"}
        missing_columns = required.difference(df.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input CSV: {sorted(missing_columns)}")

        pairs = list(zip(df["Element1"], df["Element2"]))
        print(f"Loaded {len(pairs)} pairs from: {input_csv}")

        stem = os.path.splitext(input_csv)[0]
        output_csv = f"{stem}_VERIFIED.csv"
    else:
        pairs = list(combinations(sorted(ELEMENTS), 2))
        print(f"Template CSV not found. Generated {len(pairs)} pairs from default element list.")
        output_csv = "miedema_enthalpy_filled_VERIFIED.csv"

    return pairs, output_csv


def get_value(mixing, e1, e2):
    try:
        return mixing.get_mixing_enthalpy(Element(e1), Element(e2))
    except TypeError:
        return mixing.get_mixing_enthalpy(e1, e2)


def main(download=True):
    pairs, output_csv = load_pairs()

    mixing = MixingEnthalpy(impute_nan=False)
    valid_symbols = {el.symbol for el in mixing.valid_element_list}

    elements_used = sorted(set(e for pair in pairs for e in pair))
    not_covered = [e for e in elements_used if e not in valid_symbols]

    if not_covered:
        print(f"\nElements absent from the matminer Miedema dataset: {not_covered}")
    else:
        print(
            f"\nAll {len(elements_used)} elements in the input dataset are "
            f"covered by the Miedema parameter table: {elements_used}"
        )

    results, missing = [], []

    for e1, e2 in pairs:
        if e1 not in valid_symbols or e2 not in valid_symbols:
            val = float("nan")
        else:
            val = get_value(mixing, e1, e2)

        if pd.isna(val):
            missing.append((e1, e2))
            source = NOT_FOUND_NOTE
        else:
            source = CITATION

        results.append(
            {
                "Element1": e1,
                "Element2": e2,
                "H_mix": val,
                "Source": source,
            }
        )

    out = pd.DataFrame(results)
    out.to_csv(output_csv, index=False)

    n_total = len(out)
    n_filled = out["H_mix"].notna().sum()

    print(f"\nWrote {n_total} rows to {output_csv}")
    print(f"Pairs found in reference table: {n_filled}/{n_total}")

    if missing:
        print(f"\n{len(missing)} pairs were not in the matminer/Takeuchi-Inoue table:")

        for e1, e2 in missing:
            print(f"   {e1}-{e2}")

    # Reference value comparison
    reference_checks = [
        ("Al", "Ni", -22),
        ("Ni", "Zr", -49),
        ("Cu", "Zr", -23),
        ("Al", "Ti", -30),
        ("Co", "Cr", -4),
        ("Cr", "Ni", -7),
        ("Co", "Ti", -28),
        ("Hf", "Ni", -42),
        ("Fe", "Zr", -25),
    ]

    print("\nReference value comparison:")

    for a, b, expected in reference_checks:
        if a in elements_used and b in elements_used:
            v = get_value(mixing, a, b)
            print(f"  {a}-{b}: matminer = {v}   reference ~ {expected}")

    if download:
        try:
            from google.colab import files

            files.download(output_csv)
        except ImportError:
            pass

    return output_csv


if __name__ == "__main__":
    main()
