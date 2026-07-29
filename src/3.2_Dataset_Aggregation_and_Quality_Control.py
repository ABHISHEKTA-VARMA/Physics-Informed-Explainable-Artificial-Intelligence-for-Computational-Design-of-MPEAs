import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


STEP_VERSION = "step3b_descriptor_enrichment_v1"

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

MASTER_DATASET_PATH = OUT_DIR / "MASTER_MPEA_DATASET.csv"
PROPERTY_DB_PATH = OUT_DIR / "UNIVERSAL_PROPERTY_DB.csv"

ENRICHED_OUT = OUT_DIR / "STEP3B_DESCRIPTOR_ENRICHED_DATASET.csv"
DESCRIPTOR_LIST_OUT = OUT_DIR / "STEP3B_DESCRIPTOR_LIST.txt"
METADATA_OUT = OUT_DIR / "STEP3B_DESCRIPTOR_METADATA.txt"


class DescriptorEnrichmentLayer:
    """Generate composition-based descriptors."""

    def __init__(
        self,
        df_mpea,
        df_props,
        active_threshold=0.01,
    ):
        self.df_mpea = df_mpea.copy()
        self.active_threshold = active_threshold

        self.df_props = (
            df_props.set_index("Element")
            if "Element" in df_props.columns
            else df_props
        )

        self.elem_cols = [
            column
            for column in self.df_mpea.columns
            if column.startswith("ELEM_")
        ]
        self.elements = [
            column.replace("ELEM_", "")
            for column in self.elem_cols
        ]

        self.physical_whitelist = [
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
            "poisson_ratio",
            "Pugh_ratio",
            "reduced_modulus_proxy",
            "specific_stiffness",
            "stiffness_density_index",
            "Tm_over_rho",
            "thermal_cond_specific",
            "modulus_to_melting_ratio",
        ]
        self.descriptors = [
            descriptor
            for descriptor in self.physical_whitelist
            if descriptor in self.df_props.columns
        ]

        if "is_refractory" in self.df_props.columns:
            refractory_mask = self.df_props["is_refractory"] == 1
            self.refractory_elements = set(
                self.df_props[refractory_mask].index
            )
        else:
            raise ValueError("Missing is_refractory flag.")

        self.range_targets = {
            "r",
            "atomic_volume",
            "Tm",
            "chi",
            "VEC",
        }

    def _is_range_target(self, desc_name):
        return desc_name in self.range_targets

    def generate_features(self):
        print(
            f"Generating descriptors for "
            f"{len(self.df_mpea)} alloys..."
        )

        C_matrix = self.df_mpea[self.elem_cols].values
        presence_mask = C_matrix >= self.active_threshold

        self.df_mpea["COMP_active_elements"] = np.sum(
            presence_mask,
            axis=1,
        )
        self.df_mpea["COMP_dominant_fraction"] = np.max(
            C_matrix,
            axis=1,
        )

        sum_sq_c = np.sum(C_matrix**2, axis=1)
        self.df_mpea["COMP_effective_elements"] = np.where(
            sum_sq_c > 0,
            1.0 / sum_sq_c,
            0,
        )

        ref_indices = [
            index
            for index, element in enumerate(self.elements)
            if element in self.refractory_elements
        ]

        if ref_indices:
            C_ref = C_matrix[:, ref_indices]
            C_ref = np.where(
                C_ref >= self.active_threshold,
                C_ref,
                0.0,
            )

            ref_frac = np.sum(C_ref, axis=1)
            self.df_mpea["COMP_refractory_fraction"] = ref_frac

            safe_ref_frac = np.where(ref_frac == 0, 1e-9, ref_frac)

            Tm_ref = np.array(
                [
                    self.df_props.loc[self.elements[index], "Tm"]
                    if "Tm" in self.df_props.columns
                    else 0
                    for index in ref_indices
                ]
            )
            ref_stability = np.dot(C_ref, Tm_ref) / safe_ref_frac
            self.df_mpea["COMP_refractory_stability"] = np.where(
                ref_frac == 0,
                0,
                ref_stability,
            )

            rel_C_ref = C_ref / safe_ref_frac[:, None]
            rel_C_ref_safe = np.where(rel_C_ref > 0, rel_C_ref, 1e-9)
            ref_entropy = -8.314 * np.sum(
                rel_C_ref * np.log(rel_C_ref_safe),
                axis=1,
            )
            self.df_mpea["COMP_refractory_entropy"] = np.where(
                ref_frac == 0,
                0,
                ref_entropy,
            )

        if "COMP_refractory_fraction" in self.df_mpea.columns:
            refractory_fraction = self.df_mpea[
                "COMP_refractory_fraction"
            ]

            if "DELTA_RADIUS" in self.df_mpea.columns:
                self.df_mpea["PHYS_ref_frac_x_DELTA_RADIUS"] = (
                    refractory_fraction * self.df_mpea["DELTA_RADIUS"]
                )

            if "CONFIG_ENTROPY" in self.df_mpea.columns:
                self.df_mpea["PHYS_ref_frac_x_CONFIG_ENTROPY"] = (
                    refractory_fraction * self.df_mpea["CONFIG_ENTROPY"]
                )

            if "H_MIX" in self.df_mpea.columns:
                self.df_mpea["PHYS_ref_frac_x_H_MIX"] = (
                    refractory_fraction * self.df_mpea["H_MIX"]
                )
                self.df_mpea["PHYS_ref_frac_x_abs_H_MIX"] = (
                    refractory_fraction * np.abs(self.df_mpea["H_MIX"])
                )

            if "OMEGA" in self.df_mpea.columns:
                self.df_mpea["PHYS_ref_frac_x_OMEGA"] = (
                    refractory_fraction * self.df_mpea["OMEGA"]
                )

        composition_sum = np.sum(C_matrix, axis=1)

        for descriptor in self.descriptors:
            property_vector = np.array(
                [
                    self.df_props.loc[element, descriptor]
                    if element in self.df_props.index
                    else 0
                    for element in self.elements
                ]
            )

            mean_feature = np.dot(C_matrix, property_vector)
            self.df_mpea[f"MEAN_{descriptor}"] = mean_feature

            variance = (
                np.sum(
                    C_matrix
                    * (property_vector - mean_feature[:, None]) ** 2,
                    axis=1,
                )
                / np.maximum(composition_sum, 1e-9)
            )
            std_feature = np.sqrt(variance)
            self.df_mpea[f"STD_{descriptor}"] = std_feature

            safe_mean = np.maximum(np.abs(mean_feature), 1e-6)
            self.df_mpea[f"MISMATCH_{descriptor}"] = (
                std_feature / safe_mean
            )

            if self._is_range_target(descriptor):
                alloy_properties = np.where(
                    presence_mask,
                    property_vector,
                    np.nan,
                )

                with np.errstate(all="ignore"):
                    min_values = np.nanmin(alloy_properties, axis=1)
                    max_values = np.nanmax(alloy_properties, axis=1)

                    min_values = np.nan_to_num(min_values, nan=0.0)
                    max_values = np.nan_to_num(max_values, nan=0.0)

                    self.df_mpea[f"MIN_{descriptor}"] = min_values
                    self.df_mpea[f"MAX_{descriptor}"] = max_values
                    self.df_mpea[f"RANGE_{descriptor}"] = (
                        max_values - min_values
                    )

        generated_columns = [
            column
            for column in self.df_mpea.columns
            if column.startswith(
                (
                    "MEAN_",
                    "STD_",
                    "MISMATCH_",
                    "RANGE_",
                    "MIN_",
                    "MAX_",
                    "COMP_",
                    "PHYS_",
                )
            )
        ]

        omega_interaction = "PHYS_ref_frac_x_OMEGA"
        check_columns = [
            column
            for column in generated_columns
            if column != omega_interaction
        ]

        if self.df_mpea[check_columns].isnull().values.any():
            raise ValueError("NaN detected in generated descriptors.")

        if omega_interaction in self.df_mpea.columns:
            if "OMEGA" not in self.df_mpea.columns:
                raise ValueError(
                    "OMEGA interaction exists without the OMEGA column."
                )

            expected_nan = self.df_mpea["OMEGA"].isna().to_numpy()
            actual_nan = self.df_mpea[omega_interaction].isna().to_numpy()

            if not np.array_equal(actual_nan, expected_nan):
                raise ValueError(
                    "OMEGA interaction has an unexpected NaN pattern."
                )

        if np.isinf(self.df_mpea[generated_columns].values).any():
            raise ValueError(
                "Infinite value detected in generated descriptors."
            )

        print(f"Generated {len(generated_columns)} descriptor features.")

        return self.df_mpea


def generated_descriptor_columns(df):
    return [
        column
        for column in df.columns
        if column.startswith(
            (
                "MEAN_",
                "STD_",
                "MISMATCH_",
                "RANGE_",
                "MIN_",
                "MAX_",
                "COMP_",
                "PHYS_",
            )
        )
    ]


def main():
    if IN_COLAB and (
        not MASTER_DATASET_PATH.exists()
        or not PROPERTY_DB_PATH.exists()
    ):
        print(
            "Upload MASTER_MPEA_DATASET.csv and "
            "UNIVERSAL_PROPERTY_DB.csv:"
        )
        files.upload()

        for filename in [
            "MASTER_MPEA_DATASET.csv",
            "UNIVERSAL_PROPERTY_DB.csv",
        ]:
            uploaded = Path(filename)
            if uploaded.exists():
                shutil.copyfile(uploaded, OUT_DIR / filename)

    required_files = [MASTER_DATASET_PATH, PROPERTY_DB_PATH]
    missing_files = [
        str(path)
        for path in required_files
        if not path.exists()
    ]

    if missing_files:
        raise FileNotFoundError(
            f"Missing required input files: {missing_files}"
        )

    df_mpea = pd.read_csv(MASTER_DATASET_PATH)
    df_props = pd.read_csv(PROPERTY_DB_PATH)

    layer = DescriptorEnrichmentLayer(df_mpea, df_props)
    enriched = layer.generate_features()

    descriptor_columns = generated_descriptor_columns(enriched)

    enriched.to_csv(ENRICHED_OUT, index=False)

    DESCRIPTOR_LIST_OUT.write_text(
        "\n".join(descriptor_columns),
        encoding="utf-8",
    )

    sha256 = hashlib.sha256(ENRICHED_OUT.read_bytes()).hexdigest()

    metadata_lines = [
        "Step 3B descriptor enrichment metadata",
        "",
        f"Step version: {STEP_VERSION}",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input dataset: {MASTER_DATASET_PATH}",
        f"Input property database: {PROPERTY_DB_PATH}",
        f"Output dataset: {ENRICHED_OUT}",
        f"Rows: {len(enriched)}",
        f"Columns: {len(enriched.columns)}",
        f"Generated descriptor count: {len(descriptor_columns)}",
        f"SHA256: {sha256}",
        "",
    ]

    METADATA_OUT.write_text(
        "\n".join(metadata_lines),
        encoding="utf-8",
    )

    print(f"Step version: {STEP_VERSION}")
    print(f"Input rows: {len(df_mpea)}")
    print(f"Output rows: {len(enriched)}")
    print(f"Generated descriptor count: {len(descriptor_columns)}")
    print(f"Output: {ENRICHED_OUT}")
    print(f"SHA256: {sha256}")
    print("Step 3B descriptor enrichment completed.")

    if IN_COLAB:
        zip_path = shutil.make_archive(
            "step3b_descriptor_enrichment_outputs",
            "zip",
            OUT_DIR,
        )
        print(f"Output archive written: {zip_path}")
        files.download(zip_path)


if __name__ == "__main__":
    main()
