#!/usr/bin/env python3
"""Generate all paper figures using saved results."""
import sys
sys.stdout.reconfigure(line_buffering=True)

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import seaborn as sns

FIG_DIR = os.path.expanduser('~/CongressionalGNN/figures')
RESULTS_DIR = os.path.expanduser('~/CongressionalGNN/results')
os.makedirs(FIG_DIR, exist_ok=True)

# Load results
with open(os.path.join(RESULTS_DIR, 'full_results.json')) as f:
    results = json.load(f)

CONGRESSES = [110, 111, 112, 113, 114, 115, 116]
YEARS = [2007, 2009, 2011, 2013, 2015, 2017, 2019]
TRAIN_CONGRESSES = [110, 111, 112, 113, 114]

sns.set_style('whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

# ---- Figure 1: Polarization Over Time ----
print("Figure 1: Polarization over time...", flush=True)
fig, ax = plt.subplots(figsize=(8, 5))
pol_true = [results['per_congress'][str(c)]['polarization_true'] for c in CONGRESSES]

ax.plot(YEARS, pol_true, 'ko-', linewidth=2.5, markersize=9, label='Observed Polarization', zorder=5)

# Shade train vs test
ax.axvspan(2007, 2015.5, alpha=0.05, color='green', label='Training Period')
ax.axvspan(2015.5, 2019.5, alpha=0.05, color='orange', label='Test Period')
ax.axvline(x=2015.5, color='gray', linestyle=':', alpha=0.7)

ax.annotate('Tea Party\nWave', xy=(2011, pol_true[2]), xytext=(2012.3, pol_true[2]+0.06),
            arrowprops=dict(arrowstyle='->', color='#D32F2F', lw=1.5), fontsize=10, color='#D32F2F')
ax.annotate('Trump\nEra', xy=(2017, pol_true[5]), xytext=(2018, pol_true[5]+0.06),
            arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=1.5), fontsize=10, color='#FF6F00')

ax.set_xlabel('Year (Start of Congress)')
ax.set_ylabel('Polarization Index')
ax.set_title('Congressional Polarization, 110th--116th Congress')
ax.legend(frameon=True, fontsize=9)
ax.set_xticks(YEARS)
ax.set_xticklabels([f'{y}\n({c}th)' for y, c in zip(YEARS, CONGRESSES)], fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig1_polarization.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig1_polarization.png'))
plt.close()

# ---- Figure 2: Network Visualization (scatter only, no edges) ----
print("Figure 2: Network visualization...", flush=True)
# Load data for visualization
import pandas as pd
fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for ax_idx, cnum in enumerate([110, 112, 116]):
    ax = axes[ax_idx]
    members = pd.read_csv(os.path.expanduser(f'~/CongressionalGNN/H{cnum}_members.csv'))
    members = members[members['chamber'] == 'House']
    
    nom1 = members['nominate_dim1'].fillna(0).values
    nom2 = members['nominate_dim2'].fillna(0).values
    party = members['party_code'].values
    
    dem_mask = party == 100
    rep_mask = party == 200
    other_mask = ~dem_mask & ~rep_mask
    
    ax.scatter(nom1[dem_mask], nom2[dem_mask], c='#1565C0', s=12, alpha=0.6, 
              edgecolors='white', linewidth=0.2, label='Democrat', zorder=5)
    ax.scatter(nom1[rep_mask], nom2[rep_mask], c='#C62828', s=12, alpha=0.6, 
              edgecolors='white', linewidth=0.2, label='Republican', zorder=5)
    if other_mask.sum() > 0:
        ax.scatter(nom1[other_mask], nom2[other_mask], c='#757575', s=12, alpha=0.6,
                  edgecolors='white', linewidth=0.2, label='Independent', zorder=5)
    
    year = 2007 + (cnum - 110) * 2
    ax.set_title(f'{cnum}th Congress ({year})')
    ax.set_xlabel('DW-NOMINATE Dim. 1 (Liberal--Conservative)')
    if ax_idx == 0:
        ax.set_ylabel('DW-NOMINATE Dim. 2')
    ax.set_xlim(-0.85, 1.05)
    ax.set_ylim(-0.85, 1.05)
    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.3)

axes[2].legend(loc='lower right', fontsize=8)
plt.suptitle('House Members in Ideological Space', fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig2_ideology_space.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig2_ideology_space.png'))
plt.close()

# ---- Figure 3: Model Comparison ----
print("Figure 3: Model comparison...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))

# Coalition F1
ax = axes[0]
models = ['CongressGAT', 'Logistic Reg.', 'Random Forest']
train_f1 = [
    results['train_metrics']['coalition_f1'],
    results['baselines']['logistic_regression']['train']['coalition_f1'],
    results['baselines']['random_forest']['train']['coalition_f1'],
]
test_f1 = [
    results['test_metrics']['coalition_f1'],
    results['baselines']['logistic_regression']['test']['coalition_f1'],
    results['baselines']['random_forest']['test']['coalition_f1'],
]
x = np.arange(len(models))
bars1 = ax.bar(x - 0.18, train_f1, 0.34, label='Train (110th--114th)', color='#43A047', alpha=0.85)
bars2 = ax.bar(x + 0.18, test_f1, 0.34, label='Test (115th--116th)', color='#FB8C00', alpha=0.85)
ax.set_ylabel('Macro F1 Score')
ax.set_title('Coalition Detection')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0.5, 1.05)
for bar_set in [bars1, bars2]:
    for bar in bar_set:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

# Defection AUC
ax = axes[1]
train_auc = [
    results['train_metrics']['defection_auc'],
    results['baselines']['logistic_regression']['train']['defection_auc'],
    results['baselines']['random_forest']['train']['defection_auc'],
]
test_auc = [
    results['test_metrics']['defection_auc'],
    results['baselines']['logistic_regression']['test']['defection_auc'],
    results['baselines']['random_forest']['test']['defection_auc'],
]
bars1 = ax.bar(x - 0.18, train_auc, 0.34, label='Train (110th--114th)', color='#43A047', alpha=0.85)
bars2 = ax.bar(x + 0.18, test_auc, 0.34, label='Test (115th--116th)', color='#FB8C00', alpha=0.85)
ax.set_ylabel('AUC-ROC')
ax.set_title('Party Defection Prediction')
ax.set_xticks(x)
ax.set_xticklabels(models, fontsize=9)
ax.legend(fontsize=8)
ax.set_ylim(0.4, 1.05)
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.4, linewidth=0.8)
for bar_set in [bars1, bars2]:
    for bar in bar_set:
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{bar.get_height():.2f}', ha='center', va='bottom', fontsize=8)

plt.suptitle('Model Performance Comparison', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig3_model_comparison.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig3_model_comparison.png'))
plt.close()

# ---- Figure 4: Defection Rates Over Time ----
print("Figure 4: Defection rates...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
def_traj = results['causal']['defection_trajectory']
rep_rates = [def_traj[str(c)]['republican'] for c in CONGRESSES]
dem_rates = [def_traj[str(c)]['democrat'] for c in CONGRESSES]

ax.plot(YEARS, rep_rates, 'o-', color='#C62828', linewidth=2, markersize=7, label='Republican')
ax.plot(YEARS, dem_rates, 's-', color='#1565C0', linewidth=2, markersize=7, label='Democrat')
ax.axvline(x=2015.5, color='gray', linestyle=':', alpha=0.7)
ax.set_xlabel('Year')
ax.set_ylabel('Mean Defection Rate')
ax.set_title('Party Defection Rates Over Time')
ax.legend(fontsize=9)
ax.set_xticks(YEARS)
ax.set_xticklabels([str(y) for y in YEARS], fontsize=9)

# Tea Party / Trump analysis
ax = axes[1]
pol_vals = [results['per_congress'][str(c)]['polarization_true'] for c in CONGRESSES]
colors = ['#43A047' if c in TRAIN_CONGRESSES else '#FB8C00' for c in CONGRESSES]
bars = ax.bar(range(len(CONGRESSES)), pol_vals, color=colors, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(len(CONGRESSES)))
ax.set_xticklabels([f'{c}th' for c in CONGRESSES], fontsize=9)
ax.set_ylabel('Polarization Index')
ax.set_title('Polarization by Congress')

# Annotate shifts
ax.annotate(f'Tea Party\n+{results["causal"]["tea_party"]["shift_111_112"]:.2f}',
           xy=(2, pol_vals[2]), xytext=(2.8, pol_vals[2]+0.02),
           arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')

legend_elements = [
    mpatches.Patch(facecolor='#43A047', alpha=0.85, label='Train'),
    mpatches.Patch(facecolor='#FB8C00', alpha=0.85, label='Test'),
]
ax.legend(handles=legend_elements, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig4_defection_analysis.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig4_defection_analysis.png'))
plt.close()

# ---- Figure 5: Attention Analysis ----
print("Figure 5: Attention analysis...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Same vs cross-party attention
ax = axes[0]
attn_summary = results['attention_summary']
same_attn = [attn_summary[str(c)]['mean_same_party_attention'] for c in CONGRESSES]
cross_attn = [attn_summary[str(c)]['mean_cross_party_attention'] for c in CONGRESSES]
ratio = [s/max(cr, 1e-8) for s, cr in zip(same_attn, cross_attn)]

x = np.arange(len(CONGRESSES))
width = 0.35
ax.bar(x - width/2, same_attn, width, label='Same-Party', color='#43A047', alpha=0.85)
ax.bar(x + width/2, cross_attn, width, label='Cross-Party', color='#E65100', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{c}th' for c in CONGRESSES])
ax.set_ylabel('Mean Attention Weight')
ax.set_title('GAT Attention by Party Alignment')
ax.legend(fontsize=9)

# Attention ratio
ax2 = axes[1]
ax2.plot(YEARS, ratio, 'ko-', linewidth=2, markersize=8)
ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
ax2.set_xlabel('Year')
ax2.set_ylabel('Same-Party / Cross-Party Attention Ratio')
ax2.set_title('Attention Ratio Over Time')
ax2.set_xticks(YEARS)
ax2.set_xticklabels([str(y) for y in YEARS], fontsize=9)
ax2.annotate('Ratio > 1: Model pays\nmore attention to co-partisans',
            xy=(0.05, 0.95), xycoords='axes fraction', fontsize=8, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig5_attention_analysis.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig5_attention_analysis.png'))
plt.close()

# ---- Figure 6: Causal Analysis ----
print("Figure 6: Causal analysis...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Tea Party
ax = axes[0]
tp = results['causal']['tea_party']
congs = ['110th\n(2007)', '111th\n(2009)', '112th\n(2011)']
vals = [tp['polarization_110'], tp['polarization_111'], tp['polarization_112']]
colors_tp = ['#78909C', '#78909C', '#D32F2F']
bars = ax.bar(range(3), vals, color=colors_tp, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(congs, fontsize=10)
ax.set_ylabel('Polarization Index')
ax.set_title('Tea Party Wave Effect')
# Draw arrow for the shift
ax.annotate('', xy=(2, vals[2]), xytext=(1, vals[1]),
           arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(1.5, (vals[1]+vals[2])/2 + 0.02, f'$\\Delta$ = +{tp["shift_111_112"]:.3f}', 
       ha='center', fontsize=11, color='#D32F2F', fontweight='bold')

# Trump
ax = axes[1]
tr = results['causal']['trump_era']
congs = ['114th\n(2015)', '115th\n(2017)', '116th\n(2019)']
vals = [tr['polarization_114'], tr['polarization_115'], tr['polarization_116']]
colors_tr = ['#78909C', '#FF6F00', '#FF6F00']
bars = ax.bar(range(3), vals, color=colors_tr, alpha=0.85, edgecolor='black', linewidth=0.5)
ax.set_xticks(range(3))
ax.set_xticklabels(congs, fontsize=10)
ax.set_ylabel('Polarization Index')
ax.set_title('Trump Era Effect')
ax.annotate('', xy=(1, vals[1]), xytext=(0, vals[0]),
           arrowprops=dict(arrowstyle='->', color='#FF6F00', lw=2))
ax.text(0.5, (vals[0]+vals[1])/2 + 0.02, f'$\\Delta$ = {tr["shift_114_115"]:.3f}', 
       ha='center', fontsize=11, color='#FF6F00', fontweight='bold')

plt.suptitle('Quasi-Causal Analysis of Political Shocks', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig6_causal_analysis.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig6_causal_analysis.png'))
plt.close()

# ---- Figure 7: Feature Importance ----
print("Figure 7: Feature importance...", flush=True)
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
feat_names = results['baselines']['rf_feature_importance']['feature_names']

for ax_idx, (task, task_name) in enumerate([('coalition', 'Coalition Detection'), ('defection', 'Defection Prediction')]):
    ax = axes[ax_idx]
    imp = results['baselines']['rf_feature_importance'][task]
    sorted_idx = np.argsort(imp)
    ax.barh(range(len(feat_names)), [imp[i] for i in sorted_idx], color='#1565C0', alpha=0.8)
    ax.set_yticks(range(len(feat_names)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx], fontsize=9)
    ax.set_xlabel('Feature Importance (Gini)')
    ax.set_title(f'{task_name}')

plt.suptitle('Random Forest Feature Importance', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig7_feature_importance.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig7_feature_importance.png'))
plt.close()

# ---- Figure 8: Per-Congress Performance ----
print("Figure 8: Per-congress performance...", flush=True)
fig, ax = plt.subplots(figsize=(9, 5))
coal_f1 = [results['per_congress'][str(c)]['coalition_f1'] for c in CONGRESSES]
def_auc = [results['per_congress'][str(c)]['defection_auc'] for c in CONGRESSES]

x = np.arange(len(CONGRESSES))
width = 0.35
ax.bar(x - width/2, coal_f1, width, label='Coalition F1', color='#1565C0', alpha=0.85)
ax.bar(x + width/2, def_auc, width, label='Defection AUC', color='#C62828', alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f'{c}th' for c in CONGRESSES])
ax.set_ylabel('Score')
ax.set_title('CongressGAT Performance by Congress')
ax.legend()
ax.set_ylim(0, 1.1)
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.3)

# Mark train/test
ax.axvline(x=4.5, color='gray', linestyle=':', alpha=0.7)
ax.text(2, 0.05, 'TRAIN', ha='center', fontsize=10, color='green', alpha=0.5)
ax.text(5.5, 0.05, 'TEST', ha='center', fontsize=10, color='orange', alpha=0.5)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, 'fig8_per_congress.pdf'))
plt.savefig(os.path.join(FIG_DIR, 'fig8_per_congress.png'))
plt.close()

print("All figures saved!", flush=True)
print(f"Figures in: {FIG_DIR}/", flush=True)
