import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

os.makedirs('Results', exist_ok=True)

df_ai = pd.read_csv('AI_PREDICTION_RESULTS.csv')
df_fem = pd.read_csv('ML_FEM_VALIDATION.csv')
df_topsis = pd.read_csv('TOPSIS_RESULTS.csv')
df_dft = pd.read_csv('DFT_RESULTS.csv')

df_ai_sub = df_ai[['Alloy ID', 'Predicted Hardness (HV)']]
df_fem_sub = df_fem[['Alloy ID', 'Agreement Index']]
df_topsis_sub = df_topsis[['Alloy ID', 'TOPSIS Score', 'Final Rank']]
df_dft_sub = df_dft[['Alloy ID', 'Validation Score']]

master_df = df_ai_sub.merge(df_fem_sub, on='Alloy ID') \
                     .merge(df_dft_sub, on='Alloy ID') \
                     .merge(df_topsis_sub, on='Alloy ID')

master_df = master_df.sort_values('Final Rank').reset_index(drop=True)

raw_metrics = ['Predicted Hardness (HV)', 'Agreement Index', 'Validation Score', 'TOPSIS Score']
norm_metrics = []

for col in raw_metrics:
    if col == "Validation Score":
        master_df[col + '_norm'] = master_df[col] 
    else:
        c_min, c_max = master_df[col].min(), master_df[col].max()
        if c_max > c_min:
            scaled = (master_df[col] - c_min) / (c_max - c_min)
            master_df[col + '_norm'] = 0.20 + 0.80 * scaled
        else:
            master_df[col + '_norm'] = 1.0
    norm_metrics.append(col + '_norm')

categories = ['AI\nPrediction', 'FEM\nValidation', 'DFT\nValidation', 'MCDM\nAssessment']
N = len(categories)

angles = [n / float(N) * 2 * np.pi for n in range(N)]
angles += angles[:1]

fig, ax = plt.subplots(figsize=(10.5, 9), subplot_kw=dict(polar=True), dpi=600)
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, weight='bold', size=12)
ax.tick_params(axis='x', pad=10)
ax.set_ylim(0, 1.05)

ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
ax.set_yticklabels([]) 
ax.spines['polar'].set_visible(False)
ax.grid(color='lightgrey', linestyle='--', linewidth=0.4)

colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e"]

for i, row in master_df.iterrows():
    values = row[norm_metrics].tolist()
    values += values[:1]
    
    label = f"Rank {row['Final Rank']} – {row['Alloy ID']}"
    
    ax.plot(angles, values, linewidth=3.0, linestyle='solid', label=label, color=colors[i])
    ax.fill(angles, values, color=colors[i], alpha=0.18)

plt.legend(loc='center left', bbox_to_anchor=(1.10, 0.5), frameon=False, 
           title="Alloys", title_fontproperties={'weight':'bold'})

plt.subplots_adjust(top=0.88, right=0.78)

plt.savefig('Results/Integrated_Validation_Radar.tiff', bbox_inches='tight', pad_inches=0.3, dpi=600, format='tiff')
plt.savefig('Results/Integrated_Validation_Radar.png', bbox_inches='tight', pad_inches=0.3, dpi=600)
plt.savefig('Results/Integrated_Validation_Radar.pdf', bbox_inches='tight', pad_inches=0.3)
plt.savefig('Results/Integrated_Validation_Radar.svg', bbox_inches='tight', pad_inches=0.3)
plt.close()

master_df.to_csv('Results/Radar_Input_Data.csv', index=False)

normalized_df = master_df[['Alloy ID'] + norm_metrics].copy()
normalized_df.columns = ['Alloy ID', 'AI Prediction', 'FEM Validation', 'DFT Validation', 'MCDM Assessment']
normalized_df.to_csv('Results/Radar_Normalized_Data.csv', index=False)
