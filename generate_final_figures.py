import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'Georgia']
rcParams['font.size'] = 11
rcParams['axes.labelsize'] = 12
rcParams['axes.titlesize'] = 13
rcParams['legend.fontsize'] = 9
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

RESULTS_DIR = os.path.expanduser('~/CongressionalGNN/results_final')
FIG_DIR = os.path.expanduser('~/CongressionalGNN/figures_final')
os.makedirs(FIG_DIR, exist_ok=True)

CONGRESSES = list(range(104, 119))
CONGRESS_YEARS = {
    104: '1995', 105: '1997', 106: '1999', 107: '2001', 108: '2003',
    109: '2005', 110: '2007', 111: '2009', 112: '2011', 113: '2013',
    114: '2015', 115: '2017', 116: '2019', 117: '2021', 118: '2023'
}

DEM_COLOR = '#2166AC'
REP_COLOR = '#B2182B'
CROSS_COLOR = '#7570B3'
GAT_COLOR = '#1B9E77'
BASELINE_COLORS = {'logistic_regression': '#D95F02', 'random_forest': '#7570B3', 'naive_drift': '#E7298A'}

def load_results():
    with open(os.path.join(RESULTS_DIR, 'congress_results.json')) as f:
        congress_results = json.load(f)
    with open(os.path.join(RESULTS_DIR, 'experiment_results.json')) as f:
        experiment_results = json.load(f)
    return congress_results, experiment_results

def fig1_polarization_trends(congress_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))

    congs = [str(c) for c in CONGRESSES]
    dem_coh = [congress_results[str(c)]['cohesion']['dem_cohesion'] for c in CONGRESSES]
    rep_coh = [congress_results[str(c)]['cohesion']['rep_cohesion'] for c in CONGRESSES]
    cross = [congress_results[str(c)]['cohesion']['cross_party_agreement'] for c in CONGRESSES]
    x_labels = [f"{c}\n({CONGRESS_YEARS[c]})" for c in CONGRESSES]

    ax = axes[0]
    ax.plot(range(len(CONGRESSES)), dem_coh, 'o-', color=DEM_COLOR, label='Democrats', markersize=5, linewidth=1.5)
    ax.plot(range(len(CONGRESSES)), rep_coh, 's-', color=REP_COLOR, label='Republicans', markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Intra-party Agreement Rate')
    ax.set_title('(a) Party Cohesion')
    ax.legend(frameon=False)
    ax.set_ylim(0.7, 1.0)
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(8.2, 0.72, 'Tea Party\nwave', fontsize=7, color='gray')

    ax = axes[1]
    ax.plot(range(len(CONGRESSES)), cross, 'D-', color=CROSS_COLOR, markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Cross-party Agreement Rate')
    ax.set_title('(b) Cross-party Agreement')
    ax.set_ylim(0.2, 0.65)
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.axvline(x=11, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.text(11.2, 0.22, 'Trump\nera', fontsize=7, color='gray')

    ax = axes[2]
    fiedler = [abs(congress_results[str(c)]['spectral']['fiedler_nominate_corr']) for c in CONGRESSES]
    ax.bar(range(len(CONGRESSES)), fiedler, color='#4DAF4A', alpha=0.8, edgecolor='#2D6A2E')
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('|Fiedler-NOMINATE Correlation|')
    ax.set_title('(c) Spectral-Ideological Alignment')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig1_polarization.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig1_polarization.png'))
    plt.close()
    print("  Fig 1 done.")

def fig2_defection_trends(congress_results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

    mean_def = [congress_results[str(c)]['mean_defection_rate'] for c in CONGRESSES]
    n_def = [congress_results[str(c)]['n_defectors_10pct'] for c in CONGRESSES]
    n_total = [congress_results[str(c)]['n_members'] for c in CONGRESSES]
    pct_def = [100 * n / t for n, t in zip(n_def, n_total)]

    ax = axes[0]
    ax.plot(range(len(CONGRESSES)), [100*d for d in mean_def], 'o-', color='#E66101', markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Defection Rate (%)')
    ax.set_title('(a) Average Defection Rate')
    ax.axvline(x=8, color='gray', linestyle=':', alpha=0.5)

    ax = axes[1]
    ax.bar(range(len(CONGRESSES)), pct_def, color='#5E3C99', alpha=0.8, edgecolor='#3B2561')
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([str(c) for c in CONGRESSES], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Members Exceeding 10% Threshold (%)')
    ax.set_title('(b) Share of Frequent Defectors')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig2_defection.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig2_defection.png'))
    plt.close()
    print("  Fig 2 done.")

def fig3_model_comparison(experiment_results):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    
    thresholds = [5, 10, 15, 20, 25]
    metrics_list = ['accuracy', 'f1', 'auc']
    metric_labels = ['Accuracy', 'F1 Score', 'AUC-ROC']
    
    for ax_idx, (metric, label) in enumerate(zip(metrics_list, metric_labels)):
        ax = axes[ax_idx]
        
        gat_vals = []
        lr_vals = []
        rf_vals = []
        naive_vals = []
        
        for t in thresholds:
            key = f'threshold_{t}'
            gat_vals.append(experiment_results[key]['gat_test'][metric])
            lr_vals.append(experiment_results[key]['baselines']['logistic_regression'][metric])
            rf_vals.append(experiment_results[key]['baselines']['random_forest'][metric])
            if metric != 'auc':
                naive_vals.append(experiment_results[key]['baselines']['naive_drift'][metric])
            else:
                naive_vals.append(0.5)
        
        x = np.arange(len(thresholds))
        w = 0.2
        
        ax.bar(x - 1.5*w, gat_vals, w, label='CongressGAT', color=GAT_COLOR, edgecolor='#0D5E3F')
        ax.bar(x - 0.5*w, lr_vals, w, label='Logistic Reg.', color=BASELINE_COLORS['logistic_regression'], edgecolor='#8B3E01')
        ax.bar(x + 0.5*w, rf_vals, w, label='Random Forest', color=BASELINE_COLORS['random_forest'], edgecolor='#4A469B')
        ax.bar(x + 1.5*w, naive_vals, w, label='Naive Drift', color=BASELINE_COLORS['naive_drift'], edgecolor='#A11D6B')
        
        ax.set_xticks(x)
        ax.set_xticklabels([f'{t}%' for t in thresholds])
        ax.set_xlabel('Defection Threshold')
        ax.set_ylabel(label)
        ax.set_title(f'({chr(97+ax_idx)}) {label}')
        if ax_idx == 0:
            ax.legend(frameon=False, fontsize=8)
        ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig3_model_comparison.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig3_model_comparison.png'))
    plt.close()
    print("  Fig 3 done.")

def fig4_threshold_sensitivity(experiment_results):
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    thresholds = [5, 10, 15, 20, 25]
    
    for model_name, color, marker, label in [
        ('gat_test', GAT_COLOR, 'o', 'CongressGAT'),
        ('logistic_regression', BASELINE_COLORS['logistic_regression'], 's', 'Logistic Reg.'),
        ('random_forest', BASELINE_COLORS['random_forest'], 'D', 'Random Forest'),
    ]:
        f1_vals = []
        for t in thresholds:
            key = f'threshold_{t}'
            if model_name == 'gat_test':
                f1_vals.append(experiment_results[key]['gat_test']['f1'])
            else:
                f1_vals.append(experiment_results[key]['baselines'][model_name]['f1'])
        ax.plot(thresholds, f1_vals, f'{marker}-', color=color, label=label, markersize=7, linewidth=1.5)
    
    ax.set_xlabel('Defection Threshold (%)')
    ax.set_ylabel('F1 Score')
    ax.set_title('Threshold Sensitivity Analysis')
    ax.legend(frameon=False)
    ax.set_xticks(thresholds)
    ax.set_ylim(0, 1.0)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig4_threshold_sensitivity.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig4_threshold_sensitivity.png'))
    plt.close()
    print("  Fig 4 done.")

def fig5_attention_analysis(experiment_results):
    if 'attention_analysis' not in experiment_results:
        print("  Skipping Fig 5 (no attention data)")
        return
    
    attn = experiment_results['attention_analysis']
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    
    congs = sorted([int(c) for c in attn.keys()])
    same_attn = [attn[str(c)]['same_party_attention'] for c in congs]
    cross_attn = [attn[str(c)]['cross_party_attention'] for c in congs]
    ratio = [attn[str(c)]['attention_ratio'] for c in congs]
    
    ax = axes[0]
    ax.plot(range(len(congs)), same_attn, 'o-', color=DEM_COLOR, label='Same-party', markersize=5, linewidth=1.5)
    ax.plot(range(len(congs)), cross_attn, 's-', color=REP_COLOR, label='Cross-party', markersize=5, linewidth=1.5)
    ax.set_xticks(range(len(congs)))
    ax.set_xticklabels([str(c) for c in congs], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('(a) GAT Attention by Party Alignment')
    ax.legend(frameon=False)
    
    ax = axes[1]
    ax.bar(range(len(congs)), ratio, color='#66A61E', alpha=0.8, edgecolor='#3D6312')
    ax.set_xticks(range(len(congs)))
    ax.set_xticklabels([str(c) for c in congs], rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Same/Cross Attention Ratio')
    ax.set_title('(b) Partisan Attention Ratio')
    ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig5_attention.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig5_attention.png'))
    plt.close()
    print("  Fig 5 done.")

def fig6_feature_importance(experiment_results):
    fi = experiment_results.get('threshold_10', {}).get('baselines', {}).get('feature_importance', {})
    if not fi:
        print("  Skipping Fig 6 (no feature importance data)")
        return
    
    fig, ax = plt.subplots(figsize=(7, 4))
    
    names = list(fi.keys())
    values = list(fi.values())
    sorted_idx = np.argsort(values)
    
    colors = ['#1B9E77', '#D95F02', '#7570B3', '#E7298A', '#66A61E', '#E6AB02']
    
    bars = ax.barh(range(len(names)), [values[i] for i in sorted_idx], 
                   color=[colors[i % len(colors)] for i in range(len(names))],
                   edgecolor='#333333', alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([names[i] for i in sorted_idx])
    ax.set_xlabel('Feature Importance (Random Forest)')
    ax.set_title('Feature Importance for Defection Prediction')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig6_feature_importance.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig6_feature_importance.png'))
    plt.close()
    print("  Fig 6 done.")

def fig7_causal_did(congress_results):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    
    cross = {c: congress_results[str(c)]['cohesion']['cross_party_agreement'] for c in CONGRESSES}
    def_rate = {c: congress_results[str(c)]['mean_defection_rate'] for c in CONGRESSES}
    
    pre_tea = [108, 109, 110, 111]
    post_tea = [112, 113, 114]
    
    ax = axes[0]
    pre_vals = [cross[c] for c in pre_tea]
    post_vals = [cross[c] for c in post_tea]
    
    all_c = pre_tea + post_tea
    all_v = pre_vals + post_vals
    x_pos = list(range(len(all_c)))
    
    ax.plot(x_pos[:len(pre_tea)], pre_vals, 'o-', color=DEM_COLOR, markersize=6, linewidth=1.5, label='Pre-Tea Party')
    ax.plot(x_pos[len(pre_tea):], post_vals, 's-', color=REP_COLOR, markersize=6, linewidth=1.5, label='Post-Tea Party')
    ax.axvline(x=len(pre_tea)-0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(c) for c in all_c])
    ax.set_ylabel('Cross-party Agreement')
    ax.set_title('(a) Tea Party Effect on Bipartisanship')
    ax.legend(frameon=False, fontsize=9)
    
    pre_mean = np.mean(pre_vals)
    post_mean = np.mean(post_vals)
    diff = post_mean - pre_mean
    ax.text(0.05, 0.05, f'Mean shift: {diff:+.3f}', transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    pre_trump = [112, 113, 114]
    post_trump = [115, 116, 117]
    
    ax = axes[1]
    pre_vals_t = [cross[c] for c in pre_trump]
    post_vals_t = [cross[c] for c in post_trump]
    
    all_c_t = pre_trump + post_trump
    x_pos_t = list(range(len(all_c_t)))
    
    ax.plot(x_pos_t[:len(pre_trump)], pre_vals_t, 'o-', color=DEM_COLOR, markersize=6, linewidth=1.5, label='Pre-Trump')
    ax.plot(x_pos_t[len(pre_trump):], post_vals_t, 's-', color=REP_COLOR, markersize=6, linewidth=1.5, label='Trump era')
    ax.axvline(x=len(pre_trump)-0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xticks(x_pos_t)
    ax.set_xticklabels([str(c) for c in all_c_t])
    ax.set_ylabel('Cross-party Agreement')
    ax.set_title('(b) Trump-era Effect on Bipartisanship')
    ax.legend(frameon=False, fontsize=9)
    
    pre_mean_t = np.mean(pre_vals_t)
    post_mean_t = np.mean(post_vals_t)
    diff_t = post_mean_t - pre_mean_t
    ax.text(0.05, 0.05, f'Mean shift: {diff_t:+.3f}', transform=ax.transAxes, fontsize=9,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig7_causal_did.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig7_causal_did.png'))
    plt.close()
    print("  Fig 7 done.")

def fig8_spectral_eigenvalues(congress_results):
    fig, ax = plt.subplots(figsize=(8, 5))
    
    selected = [104, 108, 112, 116, 118]
    colors_map = {104: '#1f77b4', 108: '#ff7f0e', 112: '#2ca02c', 116: '#d62728', 118: '#9467bd'}
    
    for c in selected:
        eigs = congress_results[str(c)]['spectral']['eigenvalues'][:15]
        ax.plot(range(1, len(eigs)+1), eigs, 'o-', color=colors_map[c], 
                label=f'Congress {c} ({CONGRESS_YEARS[c]})', markersize=4, linewidth=1.2)
    
    ax.set_xlabel('Eigenvalue Index')
    ax.set_ylabel('Eigenvalue (Normalized Laplacian)')
    ax.set_title('Spectral Structure of Congressional Voting Networks')
    ax.legend(frameon=False, fontsize=9)
    ax.set_xlim(0.5, 15.5)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'fig8_spectral.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'fig8_spectral.png'))
    plt.close()
    print("  Fig 8 done.")

if __name__ == '__main__':
    print("Loading results...")
    congress_results, experiment_results = load_results()
    
    print("Generating figures...")
    fig1_polarization_trends(congress_results)
    fig2_defection_trends(congress_results)
    fig3_model_comparison(experiment_results)
    fig4_threshold_sensitivity(experiment_results)
    fig5_attention_analysis(experiment_results)
    fig6_feature_importance(experiment_results)
    fig7_causal_did(congress_results)
    fig8_spectral_eigenvalues(congress_results)
    
    print(f"\nAll figures saved to {FIG_DIR}")
