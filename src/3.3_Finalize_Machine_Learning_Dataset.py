import hashlib
import shutil
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import (
    VarianceThreshold,
    mutual_info_regression,
)

try:
    from google.colab import files

    IN_COLAB = True
except ImportError:
    IN_COLAB = False


STEP_VERSION = "step3c_descriptor_screening_v1"
RANDOM_STATE = 42

OUT_DIR = Path("output")
OUT_DIR.mkdir(parents=True, exist_ok=True)

STEP3B_INPUT = OUT_DIR / "STEP3B_DESCRIPTOR_ENRICHED_DATASET.csv"

STEP3C_OUTPUT = OUT_DIR / "STEP3C_SELECTED_DESCRIPTOR_DATASET.csv"
STEP3C_FEATURES_OUT = OUT_DIR / "STEP3C_SELECTED_FEATURES.csv"
STEP3C_METADATA_OUT = OUT_DIR / "STEP3C_METADATA.txt"

RF_IMPORTANCE_OUT = OUT_DIR / "RF_IMPORTANCE_STABILITY.csv"
MI_RANKING_OUT = OUT_DIR / "MI_RANKING.csv"
FEATURE_AUDIT_OUT = OUT_DIR / "FEATURE_SELECTION_AUDIT.csv"


class DescriptorScreeningEngine:
    """Screen descriptors and export feature rankings."""

    def __init__(
        self,
        df_enriched,
        target_col="PROPERTY: HV",
        corr_threshold=0.90,
        importance_threshold=0.95,
        rf_runs=20,
    ):
        self.df = df_enriched.copy()
        self.target_col = target_col
        self.corr_threshold = corr_threshold
        self.importance_threshold = importance_threshold
        self.rf_runs = rf_runs

        # Required Step 1 features
        self.protected_features = [
            "CONFIG_ENTROPY",
            "VEC",
            "TM_AVG",
            "DELTA_RADIUS",
            "DELTA_CHI",
            "H_MIX",
            "OMEGA",
        ]

        # Candidate features
        self.feature_cols = [
            column
            for column in self.df.columns
            if column.startswith(
                (
                    "COMP_",
                    "PHYS_",
                    "MEAN_",
                    "STD_",
                    "MISMATCH_",
                    "RANGE_",
                    "MIN_",
                    "MAX_",
                )
            )
            or column in self.protected_features
        ]
        self.feature_cols = [
            column
            for column in self.feature_cols
            if column in self.df.columns
        ]

        # Audit records
        self.audit_records = {
            feature: {
                "Status": "Retained",
                "Reason": "Passed all tiers",
                "MI_Score": np.nan,
                "Mean_RF_Importance": np.nan,
            }
            for feature in self.feature_cols
        }

    def execute_pipeline(self):
        print(f"Screening {len(self.feature_cols)} features...")

        # Check required features
        print("\nProtected features:")
        for feature in self.protected_features:
            if feature in self.df.columns:
                print(f"  Available: {feature}")
            else:
                print(f"  Not found: {feature}")
        print()

        df_clean = self.df.dropna(
            subset=[self.target_col]
        ).copy()
        X = df_clean[self.feature_cols]
        y = df_clean[self.target_col]

        # Variance filter
        var_filter = VarianceThreshold(threshold=1e-5)
        var_filter.fit(X)
        surviving_tier1 = X.columns[
            var_filter.get_support()
        ].tolist()

        for feature in self.feature_cols:
            if feature not in surviving_tier1:
                self.audit_records[feature].update(
                    {
                        "Status": "Dropped",
                        "Reason": "Zero/Near-zero variance",
                    }
                )

        X = X[surviving_tier1]

        # Mutual information
        mi_scores = mutual_info_regression(
            X.fillna(0),
            y,
            random_state=RANDOM_STATE,
        )
        mi_series = pd.Series(
            mi_scores,
            index=X.columns,
        )

        for feature, mi_score in mi_series.items():
            self.audit_records[feature]["MI_Score"] = mi_score

        # Collinearity filter
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(
            np.triu(
                np.ones(corr_matrix.shape),
                k=1,
            ).astype(bool)
        )

        var_series = X.var()
        to_drop = set()

        for column in upper.columns:
            peers = upper.index[
                upper[column] > self.corr_threshold
            ].tolist()

            for peer in peers:
                if peer in to_drop or column in to_drop:
                    continue

                is_column_shielded = (
                    column in self.protected_features
                    or column.startswith("PHYS_")
                )
                is_peer_shielded = (
                    peer in self.protected_features
                    or peer.startswith("PHYS_")
                )

                if is_column_shielded and is_peer_shielded:
                    continue

                if is_column_shielded:
                    to_drop.add(peer)
                    self.audit_records[peer].update(
                        {
                            "Status": "Dropped",
                            "Reason": (
                                f"Collinear with shielded {column}"
                            ),
                        }
                    )
                elif is_peer_shielded:
                    to_drop.add(column)
                    self.audit_records[column].update(
                        {
                            "Status": "Dropped",
                            "Reason": (
                                f"Collinear with shielded {peer}"
                            ),
                        }
                    )
                    break
                else:
                    if var_series[column] > var_series[peer]:
                        to_drop.add(peer)
                        self.audit_records[peer].update(
                            {
                                "Status": "Dropped",
                                "Reason": (
                                    f"Collinear with {column}, "
                                    "lower variance"
                                ),
                            }
                        )
                    else:
                        to_drop.add(column)
                        self.audit_records[column].update(
                            {
                                "Status": "Dropped",
                                "Reason": (
                                    f"Collinear with {peer}, "
                                    "lower variance"
                                ),
                            }
                        )
                        break

        X = X[
            [
                column
                for column in X.columns
                if column not in to_drop
            ]
        ]

        # Random-forest importance
        all_importances = []

        for seed in range(self.rf_runs):
            rf = RandomForestRegressor(
                n_estimators=500,
                max_depth=10,
                min_samples_leaf=2,
                random_state=seed,
                n_jobs=-1,
            )
            rf.fit(X.fillna(0), y)
            all_importances.append(
                rf.feature_importances_
            )

        mean_imp = np.mean(all_importances, axis=0)
        std_imp = np.std(all_importances, axis=0)

        importance_series = pd.Series(
            mean_imp,
            index=X.columns,
        ).sort_values(ascending=False)

        for feature, importance in importance_series.items():
            self.audit_records[feature][
                "Mean_RF_Importance"
            ] = importance

        # Cumulative cutoff
        cumulative_importance = (
            importance_series.cumsum()
            / importance_series.sum()
        )
        cutoff_idx = np.argmax(
            cumulative_importance.values
            >= self.importance_threshold
        )
        final_features = importance_series.iloc[
            : cutoff_idx + 1
        ].index.tolist()

        # Restore required features
        for feature in X.columns:
            is_protected = (
                feature in self.protected_features
                or feature.startswith("PHYS_")
            )

            if is_protected and feature not in final_features:
                final_features.append(feature)
                self.audit_records[feature].update(
                    {
                        "Status": "Retained (Protected)",
                        "Reason": (
                            "Restored physics/cross-level core"
                        ),
                    }
                )

        # Importance stability
        safe_mean = np.where(
            mean_imp < 1e-12,
            np.nan,
            mean_imp,
        )
        importance_df = pd.DataFrame(
            {
                "Mean_Importance": mean_imp,
                "Std_Importance": std_imp,
                "CV_Importance": std_imp / safe_mean,
            },
            index=X.columns,
        ).sort_values(
            "Mean_Importance",
            ascending=False,
        )

        # Export rankings and audit
        importance_df.to_csv(RF_IMPORTANCE_OUT)

        mi_series.sort_values(
            ascending=False
        ).to_csv(
            MI_RANKING_OUT,
            header=["MI_Score"],
        )

        pd.DataFrame(
            {"Selected_Feature": final_features}
        ).to_csv(
            STEP3C_FEATURES_OUT,
            index=False,
        )

        audit_df = pd.DataFrame.from_dict(
            self.audit_records,
            orient="index",
        ).reset_index()
        audit_df.rename(
            columns={"index": "Feature"},
            inplace=True,
        )
        audit_df = audit_df[
            [
                "Feature",
                "Status",
                "Reason",
                "MI_Score",
                "Mean_RF_Importance",
            ]
        ]
        audit_df.to_csv(
            FEATURE_AUDIT_OUT,
            index=False,
        )

        retained_columns = [
            column
            for column in self.df.columns
            if column not in self.feature_cols
        ] + final_features

        return self.df[retained_columns], final_features


def main():
    if IN_COLAB and not STEP3B_INPUT.exists():
        print("Upload STEP3B_DESCRIPTOR_ENRICHED_DATASET.csv:")
        files.upload()

        uploaded = Path("STEP3B_DESCRIPTOR_ENRICHED_DATASET.csv")
        if uploaded.exists():
            shutil.copyfile(uploaded, STEP3B_INPUT)

    if not STEP3B_INPUT.exists():
        raise FileNotFoundError(
            f"Step 3B output not found: {STEP3B_INPUT}"
        )

    df_step3b = pd.read_csv(STEP3B_INPUT)

    print(f"Step version: {STEP_VERSION}")
    print(f"Loaded Step 3B dataset: {df_step3b.shape}")

    print("\nRunning Step 3C descriptor screening...\n")

    screening_engine = DescriptorScreeningEngine(
        df_step3b,
        target_col="PROPERTY: HV",
        corr_threshold=0.90,
        importance_threshold=0.95,
        rf_runs=20,
    )

    result = screening_engine.execute_pipeline()

    if isinstance(result, tuple) and len(result) == 2:
        df_step3c, final_features = result
    else:
        raise RuntimeError("Unexpected Step 3C return format")

    df_step3c.to_csv(
        STEP3C_OUTPUT,
        index=False,
    )

    dataset_sha256 = hashlib.sha256(
        STEP3C_OUTPUT.read_bytes()
    ).hexdigest()

    dropped_count = (
        len(screening_engine.feature_cols) - len(final_features)
    )

    metadata_lines = [
        "Step 3C descriptor screening metadata",
        "",
        f"Step version: {STEP_VERSION}",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        f"Input dataset: {STEP3B_INPUT}",
        f"Output dataset: {STEP3C_OUTPUT}",
        f"Input rows: {df_step3b.shape[0]}",
        f"Input columns: {df_step3b.shape[1]}",
        f"Output rows: {df_step3c.shape[0]}",
        f"Output columns: {df_step3c.shape[1]}",
        f"Candidate descriptor count: {len(screening_engine.feature_cols)}",
        f"Selected descriptor count: {len(final_features)}",
        f"Dropped descriptor count: {dropped_count}",
        f"Correlation threshold: {screening_engine.corr_threshold}",
        (
            "Cumulative importance threshold: "
            f"{screening_engine.importance_threshold}"
        ),
        f"Random forest runs: {screening_engine.rf_runs}",
        f"SHA256: {dataset_sha256}",
        "",
        f"Selected features file: {STEP3C_FEATURES_OUT}",
        f"RF importance file: {RF_IMPORTANCE_OUT}",
        f"MI ranking file: {MI_RANKING_OUT}",
        f"Feature audit file: {FEATURE_AUDIT_OUT}",
    ]

    STEP3C_METADATA_OUT.write_text(
        "\n".join(metadata_lines),
        encoding="utf-8",
    )

    print("\nVerification")
    print(f"Input: {STEP3B_INPUT} --> {STEP3B_INPUT.exists()}")
    print(f"Output: {STEP3C_OUTPUT} --> {STEP3C_OUTPUT.exists()}")
    print(
        f"Selected features: {STEP3C_FEATURES_OUT} --> "
        f"{STEP3C_FEATURES_OUT.exists()}"
    )
    print(
        f"RF importance: {RF_IMPORTANCE_OUT} --> "
        f"{RF_IMPORTANCE_OUT.exists()}"
    )
    print(
        f"MI ranking: {MI_RANKING_OUT} --> "
        f"{MI_RANKING_OUT.exists()}"
    )
    print(
        f"Feature audit: {FEATURE_AUDIT_OUT} --> "
        f"{FEATURE_AUDIT_OUT.exists()}"
    )
    print(
        f"Metadata: {STEP3C_METADATA_OUT} --> "
        f"{STEP3C_METADATA_OUT.exists()}"
    )

    print(
        f"\nFinal descriptor dataset contains "
        f"{df_step3c.shape[0]} alloys and "
        f"{df_step3c.shape[1]} columns."
    )
    print(f"Selected descriptors: {len(final_features)}")
    print(f"Dropped descriptors: {dropped_count}")
    print(f"Output path: {STEP3C_OUTPUT}")
    print(f"SHA256: {dataset_sha256}")
    print("Step 3C descriptor screening completed.")

    if IN_COLAB:
        zip_path = shutil.make_archive(
            "step3c_descriptor_screening_outputs",
            "zip",
            OUT_DIR,
        )
        print(f"Output archive written: {zip_path}")
        files.download(zip_path)


if __name__ == "__main__":
    main()
