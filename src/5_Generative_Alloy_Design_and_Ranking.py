import numpy as np
import pandas as pd
from pathlib import Path
import warnings
import joblib
import json
import hashlib
from datetime import datetime
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

try:
    from step3b_enrichment import DescriptorEnrichmentLayer
    ENRICHMENT_AVAILABLE = True
except ImportError:
    ENRICHMENT_AVAILABLE = False
    print("CRITICAL: 'step3b_enrichment.py' not found. Enrichment & Screening will fail.")

warnings.filterwarnings("ignore")

# Configuration
RANDOM_STATE = 42
TARGET_GENERATION_COUNT = 100000
BATCH_SIZE = 10000
ENRICHMENT_BATCH_SIZE = 25000
MAX_GENERATION_ATTEMPTS = 50000000

MIN_HV_THRESHOLD = 250.0
EXPORT_HV_MIN = 250.0

ACTIVE_THRESHOLD = 0.01
FAMILY_THRESHOLD = 0.05

# Sampling filters
MAX_PER_FAMILY = 25
MIN_FAMILY_OBSERVATIONS = 2

# Ranking weights
WEIGHT_HV = 0.35
WEIGHT_ENTROPY = 0.25
WEIGHT_AD_SAFETY = 0.20
WEIGHT_NOVELTY = 0.20

# Physical constraints
MIN_ELEMENT_COUNT = 4
MIN_ENTROPY_R = 1.5
MAX_DELTA_RADIUS = 8.5
MIN_H_MIX = -20.0
MAX_H_MIX = 5.0
MIN_VEC = 4.0
MAX_VEC = 9.0
MIN_OMEGA = 1.1
R_CONST = 8.314

# Paths
OUT_ROOT = Path("output")
DIR_STEP5 = OUT_ROOT / "step5_unified_generation"
DIR_STEP5.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = DIR_STEP5 / "GENERATED_ALLOYS_RANKED.csv"

MASTER_PATH = OUT_ROOT / "MASTER_MPEA_DATASET.csv"
PROPS_PATH = OUT_ROOT / "UNIVERSAL_PROPERTY_DB.csv"
MIEDEMA_PATH = "miedema_enthalpy_template_VERIFIED.csv"

AD_DIR = OUT_ROOT / "step4_applicability_domain"
AD_IMPUTER_PATH = AD_DIR / "ad_imputer.pkl"
AD_MCD_PATH = AD_DIR / "ad_covariance_model.pkl"
AD_DISTANCES_PATH = AD_DIR / "training_mahalanobis_distances.csv"

ML_DIR = OUT_ROOT / "step4_hardness_regressor"
HV_MODEL_PATH = ML_DIR / "best_hardness_model.pkl"
HV_FEATURES_PATH = ML_DIR / "hardness_features.csv"
STABLE_FEATURES_PATH = HV_FEATURES_PATH

METADATA_FILE = DIR_STEP5 / "generation_metadata.json"
MANIFEST_FILE = DIR_STEP5 / "step5_manifest.txt"

def get_family_key(row, elements):
    """Calculate the alloy family."""
    elems = [el for el in elements if row.get(f"ELEM_{el}", 0) >= FAMILY_THRESHOLD]
    return "_".join(sorted(elems))

def format_formula(row, active_elements):
    """Format the chemical formula."""
    f_str = ""
    for el in active_elements:
        val = row.get(f"ELEM_{el}", 0)
        if val >= ACTIVE_THRESHOLD:
            pct = round(val * 100, 1)
            if pct % 1 == 0:
                f_str += f"{el}{int(pct)}"
            else:
                f_str += f"{el}{pct}"
    return f_str

class UnifiedManifoldGenerator:
    def __init__(self):
        print("Step 5: generation and ranking")
        self._load_data()
        self._load_models()
        self.total_generated_raw = 0
        self.chemistry_pass_count = 0
        self.duplicate_count = 0
        self.total_ad_pass = 0
        self.sample_counter = 0

    def _load_data(self):
        print("[1] Loading data...")
        self.master_df = pd.read_csv(MASTER_PATH)
        self.props_df = pd.read_csv(PROPS_PATH).set_index("Element")

        self.elem_cols = [c for c in self.master_df.columns if c.startswith("ELEM_")]
        self.active_elements = [c.replace("ELEM_", "") for c in self.elem_cols]

        self.process_cols = [c for c in self.master_df.columns if c.startswith("PROCESS_CONDITION_") and "UNKNOWN" not in c]
        self.phase_cols = [c for c in self.master_df.columns if c.startswith("PHASE_OBSERVED_") and "UNKNOWN" not in c]

        self.r_vec = np.array([self.props_df.loc[el, "r"] if el in self.props_df.index else 1.0 for el in self.active_elements])
        self.vec_vec = np.array([self.props_df.loc[el, "VEC"] if el in self.props_df.index else 1.0 for el in self.active_elements])
        self.chi_vec = np.array([self.props_df.loc[el, "chi"] if el in self.props_df.index else 1.0 for el in self.active_elements])
        self.tm_vec = np.array([self.props_df.loc[el, "Tm"] if el in self.props_df.index else 1000.0 for el in self.active_elements])

        n_elem = len(self.active_elements)
        self.elem_to_idx = {el: i for i, el in enumerate(self.active_elements)}
        self.H_matrix = np.zeros((n_elem, n_elem))

        try:
            miedema = pd.read_csv(MIEDEMA_PATH)
            for _, row in miedema.iterrows():
                e1, e2, h = str(row["Element1"]).strip(), str(row["Element2"]).strip(), float(row["H_mix"])
                if e1 in self.elem_to_idx and e2 in self.elem_to_idx:
                    self.H_matrix[self.elem_to_idx[e1], self.elem_to_idx[e2]] = h
                    self.H_matrix[self.elem_to_idx[e2], self.elem_to_idx[e1]] = h
        except FileNotFoundError:
            pass

    def _load_models(self):
        print("[2] Loading models...")
        self.ad_imputer = joblib.load(AD_IMPUTER_PATH)
        self.ad_mcd = joblib.load(AD_MCD_PATH)

        self.hv_model = joblib.load(HV_MODEL_PATH)
        self.hv_features = pd.read_csv(HV_FEATURES_PATH)["Features"].tolist()

        if hasattr(self.hv_model, "feature_names_in_"):
            self.model_expected_features = list(self.hv_model.feature_names_in_)
        elif hasattr(self.hv_model.regressor_, "feature_names_in_"):
            self.model_expected_features = list(self.hv_model.regressor_.feature_names_in_)
        else:
            self.model_expected_features = self.hv_features

        train_md_sq = pd.read_csv(AD_DISTANCES_PATH)["Train_Mahalanobis"].values
        self.t90_sq = np.percentile(train_md_sq, 90)
        self.t95_sq = np.percentile(train_md_sq, 95)
        self.t99_sq = np.percentile(train_md_sq, 99)

    def _get_process_phase(self, row):
        active_proc = [c for c in self.process_cols if row[c] == 1]
        p_str = active_proc[0].replace("PROCESS_CONDITION_", "") if active_proc else "AS_CAST"
        active_phase = [c for c in self.phase_cols if row[c] == 1]
        ph_str = active_phase[0].replace("PHASE_OBSERVED_", "") if active_phase else "BCC"
        return p_str, ph_str

    def learn_manifold(self):
        print("[3] Learning composition manifolds...")
        np.random.seed(RANDOM_STATE)

        valid_mask = (self.master_df["PROCESS_CONDITION_UNKNOWN"] == 0) & (self.master_df["PHASE_OBSERVED_UNKNOWN"] == 0)
        valid_df = self.master_df[valid_mask].copy().reset_index(drop=True)

        valid_df["FORMULA"] = valid_df.apply(lambda r: format_formula(r, self.active_elements), axis=1)
        valid_df["ALLOY_FAMILY"] = valid_df.apply(lambda r: get_family_key(r, self.active_elements), axis=1)
        valid_df[["PROCESS", "PHASE"]] = valid_df.apply(lambda r: pd.Series(self._get_process_phase(r)), axis=1)
        valid_df["PROCESS_PHASE_SIGNATURE"] = valid_df["PROCESS"] + "_" + valid_df["PHASE"]
        valid_df["FAM_SIG"] = valid_df["ALLOY_FAMILY"] + "|" + valid_df["PROCESS_PHASE_SIGNATURE"]

        if "ALLOY_HASH" not in valid_df.columns:
            valid_df["ALLOY_HASH"] = valid_df.index.astype(str)

        self.train_info = valid_df[["ALLOY_HASH", "FORMULA", "PROCESS", "PHASE"]].to_dict('records')

        print("    Building training support indices...")
        self.train_family_support = valid_df["ALLOY_FAMILY"].value_counts().to_dict()
        self.train_process_support = valid_df["PROCESS"].value_counts().to_dict()
        self.train_phase_support = valid_df["PHASE"].value_counts().to_dict()
        self.train_fam_sig_support = valid_df["FAM_SIG"].value_counts().to_dict()

        ref_col = next((c for c in valid_df.columns if "REF" in c.upper() or "DOI" in c.upper() or "SOURCE" in c.upper()), None)
        self.family_pubs = {}
        if ref_col:
            for fam, grp in valid_df.groupby("ALLOY_FAMILY"):
                self.family_pubs[fam] = grp[ref_col].nunique()
        else:
            self.family_pubs = {fam: 1 for fam in self.train_family_support.keys()}

        if ENRICHMENT_AVAILABLE:
            train_enricher = DescriptorEnrichmentLayer(valid_df, self.props_df, active_threshold=ACTIVE_THRESHOLD)
            train_enriched = train_enricher.generate_features()
            for col in self.hv_features:
                if col not in train_enriched.columns: train_enriched[col] = 0.0
            self.train_X = train_enriched[self.hv_features].fillna(0).values.astype(float)

        C_matrix = valid_df[self.elem_cols].values
        precisions = []
        for i in range(len(self.active_elements)):
            mask = C_matrix[:, i] > 0
            if mask.sum() > 5:
                mu = C_matrix[mask, i].mean()
                var = C_matrix[mask, i].var()
                if var > 0:
                    s = (mu * (1.0 - mu) / var) - 1.0
                    if s > 0: precisions.append(s)
        self.global_precision = np.median(precisions) if precisions else 10.0

        self.sub_manifolds = []
        self.sub_manifold_probs = []
        total_valid = 0

        grouped = valid_df.groupby("ALLOY_FAMILY")
        for family, group in grouped:
            if len(group) < MIN_FAMILY_OBSERVATIONS: continue

            sig_counts = group["PROCESS_PHASE_SIGNATURE"].value_counts()

            present_elems = family.split("_")
            idx_list = [self.elem_to_idx[el] for el in present_elems]

            sig_obs_probs = (sig_counts / sig_counts.sum()).to_dict()
            sig_list = list(sig_obs_probs.keys())

            n_clusters = max(2, min(8, int(np.sqrt(len(group)))))
            n_clusters = min(n_clusters, len(group))

            X_comp = group[self.elem_cols].values[:, idx_list]

            if n_clusters > 1 and ENRICHMENT_AVAILABLE and len(group) >= 5:
                X_desc = self.train_X[group.index]
                X_scaled = StandardScaler().fit_transform(X_desc)

                try:
                    pca = PCA(n_components=0.95, random_state=RANDOM_STATE)
                    X_km = pca.fit_transform(X_scaled)
                except ValueError:
                    X_km = X_scaled

                kmeans = KMeans(n_clusters=n_clusters, random_state=RANDOM_STATE, n_init=10)
                labels = kmeans.fit_predict(X_km)
            else:
                labels = np.zeros(len(group))

            for cluster_idx in np.unique(labels):
                cluster_mask = (labels == cluster_idx)
                cluster_size = cluster_mask.sum()
                if cluster_size == 0: continue

                mu = X_comp[cluster_mask].mean(axis=0)
                mu = mu / mu.sum()
                alpha = mu * self.global_precision

                self.sub_manifolds.append({
                    "family": family,
                    "cluster_id": f"{family}_C{cluster_idx}",
                    "idx_list": idx_list,
                    "alpha": alpha,
                    "sig_list": sig_list,
                    "sig_probs": sig_obs_probs
                })
                self.sub_manifold_probs.append(cluster_size)
                total_valid += cluster_size

        self.sub_manifold_probs = np.array(self.sub_manifold_probs) / total_valid
        print(f"    Mapped {len(self.sub_manifolds)} Dirichlet sub-manifolds.")

    def get_pareto_front(self, df):
        costs = np.column_stack([
            -df["PREDICTED_HV"].values,
            -df["CONFIG_ENTROPY"].values,
            df["AD_DISTANCE_SQ"].values,
            -df["NOVELTY_COSINE"].values
        ])
        n_points = costs.shape[0]
        is_pareto = np.ones(n_points, dtype=bool)
        for i in range(n_points):
            if not is_pareto[i]: continue
            for j in range(n_points):
                if i == j: continue
                dominates = np.all(costs[j] <= costs[i]) and np.any(costs[j] < costs[i])
                if dominates:
                    is_pareto[i] = False
                    break
        return is_pareto

    def run_fps(self, df, n_keep):
        if len(df) <= n_keep: return df

        missing_cols = [c for c in self.hv_features if c not in df.columns]

        if missing_cols:
            print(f"    [FPS] Regenerating ML features for diversity sampling ({len(df)} rows)...")
            temp_df = df.copy()
            enricher = DescriptorEnrichmentLayer(temp_df, self.props_df, active_threshold=ACTIVE_THRESHOLD)
            enriched_df = enricher.generate_features()

            for col in self.model_expected_features:
                if col not in enriched_df.columns:
                    enriched_df[col] = temp_df[col].values if col in temp_df.columns else 0.0

            X_raw = enriched_df[self.hv_features].fillna(0).values.astype(float)
        else:
            X_raw = df[self.hv_features].fillna(0).values.astype(float)

        X_scaled = np.asarray(StandardScaler().fit_transform(self.ad_imputer.transform(X_raw)))

        best_rank_idx = int(np.argmax(df["RANK_SCORE"].values))
        idx = [best_rank_idx]
        dists = cdist([X_scaled[best_rank_idx]], X_scaled, metric='euclidean').flatten()

        for _ in range(1, n_keep):
            best = int(np.argmax(dists))
            idx.append(best)
            new_dists = cdist([X_scaled[best]], X_scaled, metric='euclidean').flatten()
            dists = np.minimum(dists, new_dists)

        return df.iloc[idx]

    def _process_enrichment_batch(self, raw_buffer, valid_candidates, unique_ad_survivors):
        temp_df = pd.DataFrame(raw_buffer)
        enrichment_layer = DescriptorEnrichmentLayer(temp_df, self.props_df, active_threshold=ACTIVE_THRESHOLD)
        enriched_df = enrichment_layer.generate_features()

        numeric_cols = enriched_df.select_dtypes(include=[np.number]).columns
        enriched_df[numeric_cols] = enriched_df[numeric_cols].astype(np.float32)

        for col in self.process_cols + self.phase_cols:
            if col not in enriched_df.columns:
                enriched_df[col] = temp_df[col].values if col in temp_df.columns else 0.0

        for col in self.model_expected_features:
            if col not in enriched_df.columns:
                enriched_df[col] = 0.0

        X_pred = enriched_df[self.model_expected_features].astype(np.float32)
        predictions = self.hv_model.predict(X_pred)

        enriched_df["PREDICTED_HV"] = np.clip(predictions, a_min=0, a_max=None)

        enriched_df = enriched_df[enriched_df["PREDICTED_HV"] >= MIN_HV_THRESHOLD].copy()
        if enriched_df.empty: return

        X_pred_filtered = enriched_df[self.hv_features].astype(np.float64)
        X_imp = self.ad_imputer.transform(X_pred_filtered)
        md_distances_sq = self.ad_mcd.mahalanobis(X_imp)

        enriched_df["AD_DISTANCE_SQ"] = md_distances_sq
        enriched_df["AD_DISTANCE"] = np.sqrt(md_distances_sq)

        conditions = [
            (md_distances_sq <= self.t90_sq),
            (md_distances_sq <= self.t95_sq),
            (md_distances_sq <= self.t99_sq)
        ]
        enriched_df["AD_TIER"] = np.select(conditions, ["CORE_DOMAIN", "EXTENDED_DOMAIN", "EDGE_DOMAIN"], default="OUTSIDE_DOMAIN")
        enriched_df["AD_PERCENTILE"] = np.select(conditions, [90, 95, 99], default=100)

        passed_ad_df = enriched_df[md_distances_sq <= self.t95_sq]
        self.total_ad_pass += len(passed_ad_df)

        if passed_ad_df.empty: return

        for h in passed_ad_df["ALLOY_HASH"].unique():
            unique_ad_survivors.add(h)

        valid_candidates.extend(passed_ad_df.to_dict('records'))

    def run_adaptive_generation(self):
        if not ENRICHMENT_AVAILABLE:
            raise RuntimeError("CRITICAL: 'step3b_enrichment.py' is required for AD screening.")

        print(f"[4] Generating candidates (target: {TARGET_GENERATION_COUNT} unique)...")

        valid_candidates = []
        raw_survivors_buffer = []
        seen_hashes = set()
        unique_ad_survivors = set()

        while len(unique_ad_survivors) < TARGET_GENERATION_COUNT and self.total_generated_raw < MAX_GENERATION_ATTEMPTS:

            C_batch = np.zeros((BATCH_SIZE, len(self.active_elements)))
            chosen_manifolds = np.random.choice(
                len(self.sub_manifolds), size=BATCH_SIZE, p=self.sub_manifold_probs, replace=True
            )

            unique_manifolds = np.unique(chosen_manifolds)
            for m_idx in unique_manifolds:
                rows = np.where(chosen_manifolds == m_idx)[0]
                manifold = self.sub_manifolds[m_idx]
                samples = np.random.dirichlet(manifold["alpha"], size=len(rows))
                C_batch[np.ix_(rows, manifold["idx_list"])] = samples

            c_safe = np.where(C_batch > 0, C_batch, 1e-9)
            config_entropy = -R_CONST * np.sum(C_batch * np.log(c_safe), axis=1)
            vec = np.dot(C_batch, self.vec_vec)
            r_avg = np.dot(C_batch, self.r_vec)
            delta_radius = 100.0 * np.sqrt(np.sum(C_batch * (1.0 - self.r_vec / r_avg[:, None])**2, axis=1))
            chi_avg = np.dot(C_batch, self.chi_vec)
            delta_chi = np.sqrt(np.sum(C_batch * (self.chi_vec - chi_avg[:, None])**2, axis=1))
            h_mix = 2.0 * np.sum(C_batch * (C_batch @ self.H_matrix), axis=1)
            tm_avg = np.dot(C_batch, self.tm_vec)

            # Omega calculation
            IDEAL_HMIX_THRESHOLD = 0.1

            omega_raw = (tm_avg * config_entropy) / (np.maximum(np.abs(h_mix), IDEAL_HMIX_THRESHOLD) * 1000.0)
            omega_finite = (np.abs(h_mix) >= IDEAL_HMIX_THRESHOLD).astype(int)

            omega = np.where(
                omega_finite == 1,
                omega_raw,
                np.nan
            )
            elem_counts = np.sum(C_batch >= FAMILY_THRESHOLD, axis=1)

            mask = (
                (elem_counts >= MIN_ELEMENT_COUNT) &
                (config_entropy >= MIN_ENTROPY_R * R_CONST) &
                (delta_radius <= MAX_DELTA_RADIUS) &
                (h_mix >= MIN_H_MIX) &
                (h_mix <= MAX_H_MIX) &
                (vec >= MIN_VEC) &
                (vec <= MAX_VEC) &
                ((omega_finite == 0) | (omega >= MIN_OMEGA))
            )

            self.total_generated_raw += BATCH_SIZE

            for idx in np.where(mask)[0]:
                row = C_batch[idx]
                alloy_hash = hashlib.sha256(np.round(row, 6).tobytes()).hexdigest()[:16]

                if alloy_hash in seen_hashes:
                    self.duplicate_count += 1
                    continue

                seen_hashes.add(alloy_hash)
                self.chemistry_pass_count += 1
                self.sample_counter += 1

                manifold = self.sub_manifolds[chosen_manifolds[idx]]
                sig_list = manifold["sig_list"]
                sig_probs = manifold["sig_probs"]

                p_vals = [sig_probs[s] for s in sig_list]
                p_vals = np.array(p_vals) / np.sum(p_vals)

                chosen_sig = np.random.choice(sig_list, p=p_vals)
                proc_str, phase_str = chosen_sig.split("_", 1)

                alloy_data = {
                    "SAMPLE_ID": f"GEN_{self.sample_counter:07d}",
                    "ALLOY_HASH": alloy_hash,
                    "ALLOY_FAMILY": manifold["family"],
                    "SOURCE_CLUSTER": manifold["cluster_id"],
                    "PREDICTED_PHASE": phase_str,
                    "PROCESS_PHASE_SIGNATURE": chosen_sig,
                    "SIGNATURE_OBSERVATION_PROBABILITY": sig_probs[chosen_sig],
                    "GENERATION_METHOD": "CLUSTERED_MANIFOLD_DIRICHLET",
                    "GENERATED_ALLOY_FLAG": 1,
                    "NUM_ELEMENTS": elem_counts[idx],
                    "CONFIG_ENTROPY": config_entropy[idx],
                    "VEC": vec[idx],
                    "DELTA_RADIUS": delta_radius[idx],
                    "DELTA_CHI": delta_chi[idx],
                    "H_MIX": h_mix[idx],
                    "TM_AVG": tm_avg[idx],

                    # Match the enrichment layer's expected key.
                    "OMEGA": omega[idx],
                    "OMEGA_FINITE": omega_finite[idx],

                    # Preserve the training feature count.
                    "MAX_ELEMENT": np.max(row),

                    "TEST_TEMP_BIN": 25.0,
                    "PROCESS_CONDITION_UNKNOWN": 0.0,
                    "PHASE_OBSERVED_UNKNOWN": 0.0
                }

                for c in self.process_cols: alloy_data[c] = 1 if c == f"PROCESS_CONDITION_{proc_str}" else 0
                for c in self.phase_cols: alloy_data[c] = 1 if c == f"PHASE_OBSERVED_{phase_str}" else 0
                for j, el in enumerate(self.active_elements): alloy_data[f"ELEM_{el}"] = row[j]

                raw_survivors_buffer.append(alloy_data)

            if len(raw_survivors_buffer) >= ENRICHMENT_BATCH_SIZE:
                self._process_enrichment_batch(raw_survivors_buffer, valid_candidates, unique_ad_survivors)
                raw_survivors_buffer = []

                if self.total_generated_raw % 100000 == 0:
                    print(f"    ... Sampled: {self.total_generated_raw:,} | P95 Target Hits: {len(unique_ad_survivors):,}/{TARGET_GENERATION_COUNT}")

        if len(raw_survivors_buffer) > 0:
            self._process_enrichment_batch(raw_survivors_buffer, valid_candidates, unique_ad_survivors)

        print(f"\n[5] Ranking and filtering candidates...")

        final_df = pd.DataFrame(valid_candidates)
        if final_df.empty:
            raise RuntimeError("CRITICAL: No valid alloys survived screening.")

        final_df = final_df.drop_duplicates(subset=["ALLOY_HASH"])

        print("    Calculating novelty metrics...")
        X_gen_raw = final_df[self.hv_features].fillna(0).values.astype(float)
        X_gen_imp = self.ad_imputer.transform(X_gen_raw)
        X_train_imp = self.ad_imputer.transform(self.train_X)

        dist_matrix_cos = cdist(X_gen_imp, X_train_imp, metric='cosine')
        dists_cos = dist_matrix_cos.min(axis=1)
        closest_idx_cos = dist_matrix_cos.argmin(axis=1)

        dist_matrix_euc = cdist(X_gen_imp, X_train_imp, metric='euclidean')
        dists_euc = dist_matrix_euc.min(axis=1)

        final_df["NOVELTY_COSINE"] = dists_cos
        final_df["NOVELTY_EUCLIDEAN"] = dists_euc

        final_df["NEAREST_TRAIN_ID"] = [self.train_info[i]["ALLOY_HASH"] for i in closest_idx_cos]
        final_df["NEAREST_TRAIN_FORMULA"] = [self.train_info[i]["FORMULA"] for i in closest_idx_cos]
        final_df["NEAREST_TRAIN_PROCESS"] = [self.train_info[i]["PROCESS"] for i in closest_idx_cos]
        final_df["NEAREST_TRAIN_PHASE"] = [self.train_info[i]["PHASE"] for i in closest_idx_cos]

        conds = [
            dists_cos < 0.01,
            (dists_cos >= 0.01) & (dists_cos < 0.05),
            (dists_cos >= 0.05) & (dists_cos < 0.15)
        ]
        final_df["NOVELTY_SCORE"] = np.select(conds, [0, 1, 2], default=3)

        hv_min, hv_max = final_df["PREDICTED_HV"].min(), final_df["PREDICTED_HV"].max()
        ad_min, ad_max = final_df["AD_DISTANCE_SQ"].min(), final_df["AD_DISTANCE_SQ"].max()
        ent_min, ent_max = final_df["CONFIG_ENTROPY"].min(), final_df["CONFIG_ENTROPY"].max()
        nov_min, nov_max = final_df["NOVELTY_COSINE"].min(), final_df["NOVELTY_COSINE"].max()

        norm_hv = (final_df["PREDICTED_HV"] - hv_min) / (hv_max - hv_min + 1e-9)
        norm_ad = 1.0 - ((final_df["AD_DISTANCE_SQ"] - ad_min) / (ad_max - ad_min + 1e-9))
        norm_ent = (final_df["CONFIG_ENTROPY"] - ent_min) / (ent_max - ent_min + 1e-9)
        norm_nov = (final_df["NOVELTY_COSINE"] - nov_min) / (nov_max - nov_min + 1e-9)

        final_df["RANK_SCORE"] = (WEIGHT_HV * norm_hv) + (WEIGHT_ENTROPY * norm_ent) + \
                                 (WEIGHT_AD_SAFETY * norm_ad) + (WEIGHT_NOVELTY * norm_nov)

        final_df = final_df.sort_values("RANK_SCORE", ascending=False)

        final_df = final_df.groupby("ALLOY_FAMILY").head(MAX_PER_FAMILY)
        final_df = final_df.sort_values("RANK_SCORE", ascending=False)

        final_df["PROCESS"] = final_df["PROCESS_PHASE_SIGNATURE"].apply(lambda x: x.split("_")[0])
        final_df["FAM_SIG"] = final_df["ALLOY_FAMILY"] + "|" + final_df["PROCESS_PHASE_SIGNATURE"]

        final_df["FAMILY_ALLOY_COUNT"] = final_df["ALLOY_FAMILY"].map(lambda x: self.train_family_support.get(x, 0))
        final_df["UNIQUE_PUBLICATIONS"] = final_df["ALLOY_FAMILY"].map(lambda x: self.family_pubs.get(x, 0))
        final_df["PROCESS_SUPPORT"] = final_df["PROCESS"].map(lambda x: self.train_process_support.get(x, 0))
        final_df["PHASE_SUPPORT"] = final_df["PREDICTED_PHASE"].map(lambda x: self.train_phase_support.get(x, 0))
        final_df["FAM_SIG_SUPPORT"] = final_df["FAM_SIG"].map(lambda x: self.train_fam_sig_support.get(x, 0))

        final_df["PARETO_OPTIMAL"] = False
        pareto_target = final_df.head(10000).copy()
        pareto_mask = self.get_pareto_front(pareto_target)
        final_df.loc[pareto_target.index, "PARETO_OPTIMAL"] = pareto_mask

        final_df["FORMULA"] = final_df.apply(lambda r: format_formula(r, self.active_elements), axis=1)

        export_cols = [
            "SAMPLE_ID", "ALLOY_HASH", "FORMULA", "ALLOY_FAMILY", "SOURCE_CLUSTER",
            "FAMILY_ALLOY_COUNT", "UNIQUE_PUBLICATIONS", "PROCESS_SUPPORT", "PHASE_SUPPORT", "FAM_SIG_SUPPORT",
            "PROCESS", "PREDICTED_PHASE", "PROCESS_PHASE_SIGNATURE", "SIGNATURE_OBSERVATION_PROBABILITY",
            "PREDICTED_HV", "RANK_SCORE", "PARETO_OPTIMAL", "NOVELTY_SCORE", "NOVELTY_COSINE", "NOVELTY_EUCLIDEAN",
            "NEAREST_TRAIN_ID", "NEAREST_TRAIN_FORMULA", "NEAREST_TRAIN_PROCESS", "NEAREST_TRAIN_PHASE",
            "AD_DISTANCE", "AD_DISTANCE_SQ", "AD_TIER", "NUM_ELEMENTS", "CONFIG_ENTROPY",

            # Include OMEGA and MAX_ELEMENT for CSV export.
            "VEC", "DELTA_RADIUS", "DELTA_CHI", "H_MIX", "OMEGA", "OMEGA_FINITE", "MAX_ELEMENT", "AD_PERCENTILE",

            "GENERATED_ALLOY_FLAG", "GENERATION_METHOD"
        ]

        elem_cols = [c for c in final_df.columns if c.startswith("ELEM_")]
        self.final_df = final_df[export_cols[:6] + elem_cols + self.process_cols + self.phase_cols + export_cols[6:]]

    def export(self):
        self.final_df.to_csv(OUTPUT_FILE, index=False)
        print(f"\n[6] Saving outputs to: {DIR_STEP5.name}/")

        family_process_best = self.final_df.groupby(["ALLOY_FAMILY", "PROCESS_PHASE_SIGNATURE"]).head(1)
        family_process_best = family_process_best.sort_values("RANK_SCORE", ascending=False)
        family_process_best.to_csv(DIR_STEP5 / "Top_Family_ProcessPhase.csv", index=False)

        print("    Building summary report...")
        summary_data = []
        for (proc, phase), grp in self.final_df.groupby(["PROCESS", "PREDICTED_PHASE"]):
            best_alloy = grp.sort_values("RANK_SCORE", ascending=False).iloc[0]
            summary_data.append({
                "Process": proc,
                "Phase": phase,
                "Best Family": best_alloy["ALLOY_FAMILY"],
                "Formula": best_alloy["FORMULA"],
                "HV": round(best_alloy["PREDICTED_HV"], 1),
                "Novelty Score": best_alloy["NOVELTY_SCORE"],
                "Rank Score": round(best_alloy["RANK_SCORE"], 3)
            })
        pd.DataFrame(summary_data).to_csv(DIR_STEP5 / "Summary_Top_Alloy_Per_Process_Phase.csv", index=False)

        print("    Generating top-100 exports...")
        self.final_df.sort_values("PREDICTED_HV", ascending=False).head(100).to_csv(DIR_STEP5 / "Top_100_Highest_HV.csv", index=False)
        self.final_df.sort_values("NOVELTY_COSINE", ascending=False).head(100).to_csv(DIR_STEP5 / "Top_100_Highest_Novelty.csv", index=False)
        self.final_df.sort_values("AD_DISTANCE_SQ", ascending=True).head(100).to_csv(DIR_STEP5 / "Top_100_Best_AD.csv", index=False)
        self.final_df.sort_values("RANK_SCORE", ascending=False).head(100).to_csv(DIR_STEP5 / "Top_100_Balanced.csv", index=False)

        print("    Running farthest point sampling (FPS)...")
        fps_500 = self.run_fps(family_process_best, min(500, len(family_process_best)))
        fps_500.to_csv(DIR_STEP5 / "Top_500_Diverse_Global.csv", index=False)

        pareto_df = self.final_df[self.final_df["PARETO_OPTIMAL"] == True]
        pareto_fps = self.run_fps(pareto_df, min(100, len(pareto_df)))
        pareto_fps.to_csv(DIR_STEP5 / "Top_100_Pareto_Diverse.csv", index=False)

        coverage_data = []
        for sig, grp in self.final_df.groupby("PROCESS_PHASE_SIGNATURE"):
            families = grp["ALLOY_FAMILY"].nunique()
            candidates = len(grp)
            phase = grp.iloc[0]["PREDICTED_PHASE"]
            proc = sig.replace(f"_{phase}", "")
            coverage_data.append({"Process": proc, "Phase": phase, "Families": families, "Candidates": candidates})

        pd.DataFrame(coverage_data).to_csv(DIR_STEP5 / "Process_Phase_Coverage.csv", index=False)

        novel_df = family_process_best[family_process_best["NOVELTY_SCORE"] >= 2]
        novel_df.to_csv(DIR_STEP5 / "Novel_Family_Representatives.csv", index=False)

        family_process_best.head(100).to_csv(DIR_STEP5 / "QE_Validation_Stack.csv", index=False)
        family_process_best.head(100).to_csv(DIR_STEP5 / "FEM_Validation_Stack.csv", index=False)

        for sig, grp in self.final_df.groupby("PROCESS_PHASE_SIGNATURE"):
            safe_sig = str(sig).replace("/", "_").replace("+", "_").replace(" ", "_")
            grp_unique = grp.groupby("ALLOY_FAMILY").head(1).sort_values("RANK_SCORE", ascending=False)
            grp_unique.head(10).to_csv(DIR_STEP5 / f"TOP10_SIG_{safe_sig}.csv", index=False)

        for phase, grp in self.final_df.groupby("PREDICTED_PHASE"):
            safe_phase = str(phase).replace("/", "_").replace("+", "_").replace(" ", "_")
            grp_unique = grp.groupby("ALLOY_FAMILY").head(1).sort_values("RANK_SCORE", ascending=False)
            grp_unique.head(10).to_csv(DIR_STEP5 / f"TOP10_PHASE_{safe_phase}.csv", index=False)

        for c in self.process_cols:
            proc = c.replace("PROCESS_CONDITION_", "")
            safe_proc = proc.replace("/", "_")
            grp = self.final_df[self.final_df[c] == 1]
            if not grp.empty:
                grp_unique = grp.groupby("ALLOY_FAMILY").head(1).sort_values("RANK_SCORE", ascending=False)
                grp_unique.head(10).to_csv(DIR_STEP5 / f"TOP10_PROCESS_{safe_proc}.csv", index=False)

        total_valid_chem = max(self.chemistry_pass_count + self.duplicate_count, 1)
        duplicate_rate_pct = (self.duplicate_count / total_valid_chem) * 100

        metadata = {
            "generator": "Unified Target-Manifold Optimizer (General MPEA)",
            "min_family_observations": MIN_FAMILY_OBSERVATIONS,
            "max_alloys_per_family": MAX_PER_FAMILY,
            "family_threshold_at_pct": FAMILY_THRESHOLD,
            "entropy_threshold": MIN_ENTROPY_R,
            "delta_radius_max": MAX_DELTA_RADIUS,
            "min_omega_param": MIN_OMEGA,
            "min_hv_prefilter": MIN_HV_THRESHOLD,
            "publication_hv_filter": EXPORT_HV_MIN,
            "ad_threshold": "P95 (EXTENDED_DOMAIN)",
            "ranking_weights": {"Hardness": WEIGHT_HV, "Entropy": WEIGHT_ENTROPY, "AD_Safety": WEIGHT_AD_SAFETY, "Novelty": WEIGHT_NOVELTY},
            "sampling_attempts_raw": self.total_generated_raw,
            "chemistry_pass_unique": self.chemistry_pass_count,
            "duplicate_rate_pct": round(duplicate_rate_pct, 4),
            "total_p95_ad_pass_unique": self.total_ad_pass,
            "final_diverse_candidates": len(self.final_df),
            "pareto_front_count": len(pareto_df),
            "unique_families_discovered": len(self.final_df["ALLOY_FAMILY"].unique()),
            "max_predicted_hv": round(self.final_df["PREDICTED_HV"].max(), 2),
            "median_predicted_hv": round(self.final_df["PREDICTED_HV"].median(), 2)
        }

        with open(METADATA_FILE, "w") as f: json.dump(metadata, f, indent=4)

        def hash_file(fp):
            if not Path(fp).exists(): return "FILE_NOT_FOUND"
            h = hashlib.sha256()
            with open(fp, 'rb') as file:
                while chunk := file.read(8192): h.update(chunk)
            return h.hexdigest()

        with open(MANIFEST_FILE, "w") as f:
            f.write("=========================================\n")
            f.write("STEP 5 UNIFIED GENERATION MANIFEST\n")
            f.write("=========================================\n")
            f.write(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Random Seed: {RANDOM_STATE}\n\n")
            f.write("INPUT HASHES (SHA-256):\n")
            f.write(f"MASTER_DATASET: {hash_file(MASTER_PATH)}\n")
            f.write(f"STABLE_FEATURES: {hash_file(STABLE_FEATURES_PATH)}\n")
            f.write(f"AD_COVARIANCE: {hash_file(AD_MCD_PATH)}\n")
            f.write(f"HV_MODEL: {hash_file(HV_MODEL_PATH)}\n\n")
            f.write("OUTPUT HASHES (SHA-256):\n")
            f.write(f"GENERATED_ALLOYS: {hash_file(OUTPUT_FILE)}\n")
            f.write(f"METADATA: {hash_file(METADATA_FILE)}\n")

        print(f"    Ranked database -> {OUTPUT_FILE.name}")
        print("\nTop 5 diverse alloy families")

        top_5 = fps_500.head(5)
        for _, row in top_5.iterrows():
            formula = row['FORMULA']
            hv = row['PREDICTED_HV']
            phase = row['PREDICTED_PHASE']
            process = row['PROCESS_PHASE_SIGNATURE'].replace(f"_{phase}", "")
            pubs = row['UNIQUE_PUBLICATIONS']
            ref = row['NEAREST_TRAIN_FORMULA'][:12] + "..." if len(row['NEAREST_TRAIN_FORMULA']) > 12 else row['NEAREST_TRAIN_FORMULA']

            # Use the exported OMEGA column.
            omega = row['OMEGA']

            print(f"Formula: {formula:<18} | HV: {hv:>5.1f} | Omega: {omega:>4.1f} | Pubs: {pubs:>3} | Phase: {phase:<6} | Process: {process:<12} | Nearest: {ref}")
        print()

        print("\nGeneration completed.")
        print(f"Enrichment descriptors : {len([c for c in self.final_df.columns if c.startswith(('COMP_','MEAN_','STD_','MISMATCH_','RANGE_','MIN_','MAX_','PHYS_'))])}")
        print(f"Hardness model features : {len(self.hv_features)}")
        print(f"Ranked alloys saved : {len(self.final_df)}")

if __name__ == "__main__":
    generator = UnifiedManifoldGenerator()
    generator.learn_manifold()
    generator.run_adaptive_generation()
    generator.export()
