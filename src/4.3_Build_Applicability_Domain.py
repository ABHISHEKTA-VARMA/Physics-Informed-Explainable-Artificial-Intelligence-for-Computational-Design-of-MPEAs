import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.covariance import MinCovDet
from sklearn import set_config

# Preserve pipeline output format.
set_config(transform_output="pandas")

# Paths
out_root = Path("output")
DIR_HARDNESS = out_root / "step4_hardness_regressor"
AD_DIR = out_root / "step4_applicability_domain"
AD_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = out_root / "STEP3C_SELECTED_DESCRIPTOR_DATASET.csv"
RANDOM_STATE = 42

print("Loading model and data...")

# Load model
best_model = joblib.load(DIR_HARDNESS / "best_hardness_model.pkl")

# Selected features
final_selected_features = best_model.regressor_.named_steps["selector"].selected_features_

# Save features
pd.DataFrame({"Features": final_selected_features}).to_csv(
    DIR_HARDNESS / "hardness_features.csv", index=False
)

# Load data
df = pd.read_csv(DATA_PATH)
TARGET_REG = "PROPERTY: HV"
df = df.dropna(subset=[TARGET_REG]).reset_index(drop=True)

X_train_ad = df[final_selected_features].astype(float)

print(f"Building applicability domain models using {len(final_selected_features)} features...")

# Applicability domain pipeline
ad_imputer = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=RANDOM_STATE))
])

X_imp = ad_imputer.fit_transform(X_train_ad)

# Fit covariance model
ad_mcd = MinCovDet(random_state=RANDOM_STATE).fit(X_imp)
train_mahalanobis = ad_mcd.mahalanobis(X_imp)

# Save outputs
joblib.dump(ad_imputer, AD_DIR / "ad_imputer.pkl")
joblib.dump(ad_mcd, AD_DIR / "ad_covariance_model.pkl")
pd.DataFrame({"Train_Mahalanobis": train_mahalanobis}).to_csv(
    AD_DIR / "training_mahalanobis_distances.csv", index=False
)

print("Applicability domain models and features exported for Step 5.")
