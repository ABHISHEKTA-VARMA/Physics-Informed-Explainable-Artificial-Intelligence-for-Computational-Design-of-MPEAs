import re
from itertools import combinations

import pandas as pd

# Read data
df_raw = pd.read_csv("MPEA_dataset.csv")

# Extract elements from formulas
dataset_elements = set()
for formula in df_raw["FORMULA"].dropna():
    elements = re.findall(r"[A-Z][a-z]?", formula)
    dataset_elements.update(elements)

# Sort elements
updated_elements = sorted(list(dataset_elements))

print(f"Elements found ({len(updated_elements)}): {updated_elements}")

# Generate binary pairs
pairs = []
for e1, e2 in combinations(updated_elements, 2):
    pairs.append(
        {
            "Element1": e1,
            "Element2": e2,
            "H_mix": "",
            "Source": "",
        }
    )

# Export results
df = pd.DataFrame(pairs)
print(f"Total binary pairs generated = {len(df)}")

df.to_csv("miedema_enthalpy_template.csv", index=False)
print("Saved: miedema_enthalpy_template.csv")
