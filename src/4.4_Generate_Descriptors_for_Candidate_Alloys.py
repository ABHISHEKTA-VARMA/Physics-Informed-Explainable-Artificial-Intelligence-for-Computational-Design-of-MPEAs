%%writefile step3b_enrichment.py

import pandas as pd
import numpy as np

class DescriptorEnrichmentLayer:
    """Generate descriptor enrichment features."""

    def __init__(self, df_mpea, df_props, active_threshold=0.01):
        self.df_mpea = df_mpea.copy()
        self.active_threshold = active_threshold

        # Property data
        self.df_props = df_props.set_index('Element') if 'Element' in df_props.columns else df_props

        # Composition columns
        self.elem_cols = [col for col in self.df_mpea.columns if col.startswith('ELEM_')]
        self.elements = [col.replace('ELEM_', '') for col in self.elem_cols]

        # Physical descriptors
        self.physical_whitelist = [
            "atomic_mass", "r", "atomic_volume", "rho", "Tm", "E", "G", "K", "VEC",
            "thermal_cond", "cohesive_energy", "work_function", "chi", "poisson_ratio",
            "Pugh_ratio", "reduced_modulus_proxy", "specific_stiffness",
            "stiffness_density_index", "Tm_over_rho", "thermal_cond_specific",
            "modulus_to_melting_ratio"
        ]
        self.descriptors = [d for d in self.physical_whitelist if d in self.df_props.columns]

        # Refractory elements
        if 'is_refractory' in self.df_props.columns:
            refractory_mask = self.df_props['is_refractory'] == 1
            self.refractory_elements = set(self.df_props[refractory_mask].index)
        else:
            raise ValueError("Missing is_refractory flag.")

        # Range descriptors
        self.range_targets = {"r", "atomic_volume", "Tm", "chi", "VEC"}

    def _is_range_target(self, desc_name):
        return desc_name in self.range_targets

    def generate_features(self):
        print(f"Enriching {len(self.df_mpea)} alloys...")

        C_matrix = self.df_mpea[self.elem_cols].values

        # Ignore trace elements.
        presence_mask = C_matrix >= self.active_threshold

        # Composition descriptors
        self.df_mpea['COMP_active_elements'] = np.sum(presence_mask, axis=1)
        self.df_mpea['COMP_dominant_fraction'] = np.max(C_matrix, axis=1)

        # Effective elements
        sum_sq_c = np.sum(C_matrix**2, axis=1)
        self.df_mpea['COMP_effective_elements'] = np.where(sum_sq_c > 0, 1.0 / sum_sq_c, 0)

        # Refractory subsystem
        ref_indices = [i for i, el in enumerate(self.elements) if el in self.refractory_elements]

        if ref_indices:
            C_ref = C_matrix[:, ref_indices]

            # Apply active threshold.
            C_ref = np.where(C_ref >= self.active_threshold, C_ref, 0.0)

            ref_frac = np.sum(C_ref, axis=1)
            self.df_mpea['COMP_refractory_fraction'] = ref_frac

            safe_ref_frac = np.where(ref_frac == 0, 1e-9, ref_frac)

            # Refractory stability
            Tm_ref = np.array([self.df_props.loc[self.elements[i], 'Tm'] if 'Tm' in self.df_props.columns else 0 for i in ref_indices])
            ref_stability = np.dot(C_ref, Tm_ref) / safe_ref_frac
            self.df_mpea['COMP_refractory_stability'] = np.where(ref_frac == 0, 0, ref_stability)

            # Refractory entropy
            rel_C_ref = C_ref / safe_ref_frac[:, None]
            rel_C_ref_safe = np.clip(rel_C_ref, 1e-9, None)
            ref_entropy = -8.314 * np.sum(rel_C_ref * np.log(rel_C_ref_safe), axis=1)
            self.df_mpea['COMP_refractory_entropy'] = np.where(ref_frac == 0, 0, ref_entropy)

        # Cross-level descriptors
        if 'COMP_refractory_fraction' in self.df_mpea.columns:
            rf = self.df_mpea['COMP_refractory_fraction']

            if 'DELTA_RADIUS' in self.df_mpea.columns:
                self.df_mpea['PHYS_ref_frac_x_DELTA_RADIUS'] = rf * self.df_mpea['DELTA_RADIUS']

            if 'CONFIG_ENTROPY' in self.df_mpea.columns:
                self.df_mpea['PHYS_ref_frac_x_CONFIG_ENTROPY'] = rf * self.df_mpea['CONFIG_ENTROPY']

            if 'H_MIX' in self.df_mpea.columns:
                self.df_mpea['PHYS_ref_frac_x_H_MIX'] = rf * self.df_mpea['H_MIX']
                self.df_mpea['PHYS_ref_frac_x_abs_H_MIX'] = rf * np.abs(self.df_mpea['H_MIX'])

            if 'OMEGA' in self.df_mpea.columns:
                self.df_mpea['PHYS_ref_frac_x_OMEGA'] = rf * self.df_mpea['OMEGA']

        # Property statistics
        composition_sum = np.maximum(np.sum(C_matrix, axis=1), 1e-9)

        for desc in self.descriptors:
            P_vector = np.array([self.df_props.loc[el, desc] if el in self.df_props.index else 0
                                 for el in self.elements])

            mean_feature = np.dot(C_matrix, P_vector) / composition_sum
            self.df_mpea[f'MEAN_{desc}'] = mean_feature

            variance = (
                np.sum(C_matrix * (P_vector - mean_feature[:, None])**2, axis=1)
                / composition_sum
            )
            std_feature = np.sqrt(variance)
            self.df_mpea[f'STD_{desc}'] = std_feature

            safe_mean = np.maximum(np.abs(mean_feature), 1e-6)
            self.df_mpea[f'MISMATCH_{desc}'] = std_feature / safe_mean

            if self._is_range_target(desc):
                alloy_props = np.where(presence_mask, P_vector, np.nan)
                with np.errstate(all='ignore'):
                    min_vals = np.nanmin(alloy_props, axis=1)
                    max_vals = np.nanmax(alloy_props, axis=1)

                    min_vals = np.nan_to_num(min_vals, nan=0.0)
                    max_vals = np.nan_to_num(max_vals, nan=0.0)

                    self.df_mpea[f'MIN_{desc}'] = min_vals
                    self.df_mpea[f'MAX_{desc}'] = max_vals
                    self.df_mpea[f'RANGE_{desc}'] = max_vals - min_vals

        # Descriptor check
        generated_cols = [
            c for c in self.df_mpea.columns
            if c.startswith((
                "MEAN_", "STD_", "MISMATCH_", "RANGE_", "MIN_", "MAX_", "COMP_", "PHYS_"
            ))
        ]

        if np.isinf(self.df_mpea[generated_cols].values).any():
            raise ValueError("Infinite value detected in generated descriptors.")

        total_features = len(generated_cols)
        print(f"Generated {total_features} enrichment features.")
        return self.df_mpea
