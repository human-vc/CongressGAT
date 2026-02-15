#!/usr/bin/env python3

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman', 'DejaVu Serif', 'serif']
rcParams['font.size'] = 10
rcParams['axes.titlesize'] = 12
rcParams['axes.labelsize'] = 11
rcParams['xtick.labelsize'] = 9
rcParams['ytick.labelsize'] = 9
rcParams['legend.fontsize'] = 9
rcParams['figure.dpi'] = 300
rcParams['savefig.dpi'] = 300
rcParams['savefig.bbox'] = 'tight'
rcParams['axes.spines.top'] = False
rcParams['axes.spines.right'] = False

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline_results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "paper_figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
    all_results = json.load(f)

with open(os.path.join(MODEL_DIR, "results.json")) as f:
    model_results = json.load(f)

with open(os.path.join(MODEL_DIR, "baseline_results.json")) as f:
    baseline_results = json.load(f)

with open(os.path.join(MODEL_DIR, "sensitivity_results.json")) as f:
    sensitivity_results = json.load(f)

with open(os.path.join(MODEL_DIR, "causal_results.json")) as f:
    causal_results = json.load(f)

CONGRESS_YEARS = {
    100: 1987, 101: 1989, 102: 1991, 103: 1993, 104: 1995, 105: 1997,
    106: 1999, 107: 2001, 108: 2003, 109: 2005, 110: 2007, 111: 2009,
    112: 2011, 113: 2013, 114: 2015, 115: 2017, 116: 2019, 117: 2021, 118: 2023,
}

DEM_COLOR = '#2166ac'
REP_COLOR = '#b2182b'
NEUTRAL_COLOR = '#636363'
ACCENT_COLOR = '#d95f02'


def draw_network_panel(ax, congress, title):
    data = np.load(os.path.join(RESULTS_DIR, f"congress_{congress}.npz"))
    with open(os.path.join(RESULTS_DIR, f"member_info_{congress}.json")) as f:
        member_info = json.load(f)

    agreement = data['agreement']
    member_list = data['member_list']
    n = len(member_list)

    parties = np.array([member_info.get(str(int(m)), {}).get('party', 0) for m in member_list])
    nom1 = np.array([member_info.get(str(int(m)), {}).get('nominate_dim1', 0.0) for m in member_list])
    nom2 = np.array([member_info.get(str(int(m)), {}).get('nominate_dim2', 0.0) for m in member_list])

    adj = (agreement > 0.65).astype(float)
    np.fill_diagonal(adj, 0)

    cross_edges = []
    same_edges = []
    for i in range(n):
        for j in range(i+1, n):
            if adj[i, j] > 0:
                if parties[i] != parties[j]:
                    cross_edges.append((i, j))
                else:
                    same_edges.append((i, j))

    max_same = 2000
    max_cross = 500
    if len(same_edges) > max_same:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(same_edges), max_same, replace=False)
        same_edges = [same_edges[k] for k in idx]
    if len(cross_edges) > max_cross:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(cross_edges), max_cross, replace=False)
        cross_edges = [cross_edges[k] for k in idx]

    from matplotlib.collections import LineCollection
    same_segs = [[(nom1[i], nom2[i]), (nom1[j], nom2[j])] for i, j in same_edges]
    cross_segs = [[(nom1[i], nom2[i]), (nom1[j], nom2[j])] for i, j in cross_edges]

    if same_segs:
        lc = LineCollection(same_segs, colors='#bbbbbb', alpha=0.08, linewidths=0.3, zorder=0)
        ax.add_collection(lc)
    if cross_segs:
        lc = LineCollection(cross_segs, colors='#e41a1c', alpha=0.25, linewidths=0.4, zorder=1)
        ax.add_collection(lc)

    dem = parties == 100
    rep = parties == 200
    ax.scatter(nom1[dem], nom2[dem], c=DEM_COLOR, s=14, alpha=0.7, zorder=3, label='Democrat')
    ax.scatter(nom1[rep], nom2[rep], c=REP_COLOR, s=14, alpha=0.7, zorder=3, label='Republican')
    ax.set_xlabel('DW-NOMINATE Dim 1 (Liberal-Conservative)')
    ax.set_ylabel('DW-NOMINATE Dim 2')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='upper left', frameon=True, framealpha=0.9)
    ax.set_xlim(-1.1, 1.1)
    n_cross = len([1 for i, j in [(i,j) for i in range(n) for j in range(i+1,n) if adj[i,j]>0] if parties[i]!=parties[j]]) if n < 100 else len(cross_edges)
    ax.text(1.0, 0.04, f'Cross-party edges: {len(cross_edges)}', transform=ax.transAxes,
           ha='right', va='bottom', fontsize=8, fontweight='bold', color='black')


def figure1_network():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    draw_network_panel(axes[1], 103, '103rd Congress (1993-1995)')
    draw_network_panel(axes[0], 114, '114th Congress (2015-2017)')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig1_network.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig1_network.png"))
    plt.close()
    print("Figure 1 saved")


def figure2_polarization():
    congresses = sorted([int(k) for k in all_results.keys()])
    years = [CONGRESS_YEARS.get(c, 1987 + (c-100)*2) for c in congresses]
    fiedler = [all_results[str(c)]['spectral']['fiedler'] for c in congresses]
    distance = [all_results[str(c)]['polarization']['party_distance'] for c in congresses]

    fig, ax1 = plt.subplots(figsize=(10, 5))

    color1 = '#1b7837'
    color2 = '#762a83'

    ax1.plot(years, fiedler, 'o-', color=color1, linewidth=2, markersize=6, label='Fiedler Value (Network Connectivity)')
    ax1.set_xlabel('Year')
    ax1.set_ylabel('Fiedler Value', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim(-0.05, max(fiedler) * 1.15)

    ax2 = ax1.twinx()
    ax2.plot(years, distance, 's-', color=color2, linewidth=2, markersize=6, label='DW-NOMINATE Party Distance')
    ax2.set_ylabel('Party Distance', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)

    events = {
        1994: 'Contract with\nAmerica',
        2001: '9/11',
        2010: 'Tea Party\nWave',
        2016: 'Trump\nElected',
    }

    for year, label in events.items():
        if min(years) <= year <= max(years):
            ax1.axvline(x=year, color='#999999', linestyle='--', alpha=0.6, linewidth=0.8)
            ax1.annotate(label, xy=(year, max(fiedler)*1.05), fontsize=7.5,
                        ha='center', va='bottom', color='#555555')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right', frameon=True, framealpha=0.9)

    ax1.set_title('Congressional Polarization: Network Structure vs. Ideology', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig2_polarization.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig2_polarization.png"))
    plt.close()
    print("Figure 2 saved")


def figure3_attention():
    target_congress = 115
    attn_path = os.path.join(MODEL_DIR, f"attention_{target_congress}.npz")
    if not os.path.exists(attn_path):
        print("Skipping Figure 3: no attention data")
        return

    attn_data = np.load(attn_path)
    with open(os.path.join(RESULTS_DIR, f"member_info_{target_congress}.json")) as f:
        member_info = json.load(f)
    data = np.load(os.path.join(RESULTS_DIR, f"congress_{target_congress}.npz"))
    member_list = data['member_list']

    attn = attn_data['layer1']
    mean_attn = attn.mean(axis=0)

    parties = np.array([
        member_info.get(str(int(m)), {}).get('party', 0) for m in member_list
    ])
    nom1 = np.array([
        member_info.get(str(int(m)), {}).get('nominate_dim1', 0.0) for m in member_list
    ])

    sort_idx = np.argsort(nom1)
    n_show = min(50, len(member_list))
    step = max(1, len(sort_idx) // n_show)
    show_idx = sort_idx[::step][:n_show]

    attn_sub = mean_attn[np.ix_(show_idx, show_idx)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5),
                              gridspec_kw={'width_ratios': [1, 1]})

    ax = axes[0]
    im = ax.imshow(attn_sub, cmap='YlOrRd', aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.8, label='Attention Weight')
    ax.set_title('GAT Attention Weights\n(Sorted by Ideology)', fontweight='bold')
    ax.set_xlabel('Representative Index (Left to Right)')
    ax.set_ylabel('Representative Index (Left to Right)')

    party_sub = parties[show_idx]
    for i in range(len(show_idx)):
        color = DEM_COLOR if party_sub[i] == 100 else REP_COLOR
        ax.plot(-1.5, i, 's', color=color, markersize=3, clip_on=False)
        ax.plot(i, -1.5, 's', color=color, markersize=3, clip_on=False)

    ax = axes[1]

    cross_party_attn = []
    same_party_attn = []
    for i in range(len(member_list)):
        for j in range(len(member_list)):
            if i == j:
                continue
            if mean_attn[i, j] > 0:
                if parties[i] != parties[j]:
                    cross_party_attn.append(mean_attn[i, j])
                else:
                    same_party_attn.append(mean_attn[i, j])

    if cross_party_attn and same_party_attn:
        bins = np.linspace(0, max(max(cross_party_attn), max(same_party_attn)) * 1.01, 50)
        ax.hist(same_party_attn, bins=bins, alpha=0.6, color='#4daf4a', label='Within Party', density=True)
        ax.hist(cross_party_attn, bins=bins, alpha=0.6, color='#e41a1c', label='Cross Party', density=True)
        ax.set_xlabel('Attention Weight')
        ax.set_ylabel('Density')
        ax.set_title('Attention Distribution by Party Alignment', fontweight='bold')
        ax.legend(frameon=True)

    # Force both subplots to the same height/aspect ratio
    for a in axes:
        a.set_box_aspect(1)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig3_attention.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig3_attention.png"))
    plt.close()
    print("Figure 3 saved")


def figure4_roc():
    fig, ax = plt.subplots(figsize=(7, 6))

    test_congresses = [115, 116, 117]
    colors = ['#e41a1c', '#377eb8', '#4daf4a']

    for i, congress in enumerate(test_congresses):
        ckey = str(congress)
        if ckey not in model_results['defection']:
            continue
        r = model_results['defection'][ckey]
        labels = np.array(r['labels'])
        preds = np.array(r['predictions'])

        if labels.sum() == 0 or labels.sum() == len(labels):
            continue

        fpr, tpr, _ = roc_curve(labels, preds)
        roc_auc = auc(fpr, tpr)
        year = CONGRESS_YEARS.get(congress, congress)
        ax.plot(fpr, tpr, color=colors[i], linewidth=2,
               label=f'{congress}th Congress ({year}): AUC = {roc_auc:.3f}')

    if 'predictions' in baseline_results:
        bl_labels = np.array(baseline_results['predictions']['test_labels'])
        lr_preds = np.array(baseline_results['predictions']['lr'])
        rf_preds = np.array(baseline_results['predictions']['rf'])

        fpr_lr, tpr_lr, _ = roc_curve(bl_labels, lr_preds)
        fpr_rf, tpr_rf, _ = roc_curve(bl_labels, rf_preds)
        ax.plot(fpr_lr, tpr_lr, color='#984ea3', linewidth=1.5, linestyle='--',
               label=f'Logistic Regression: AUC = {auc(fpr_lr, tpr_lr):.3f}')
        ax.plot(fpr_rf, tpr_rf, color='#ff7f00', linewidth=1.5, linestyle='--',
               label=f'Random Forest: AUC = {auc(fpr_rf, tpr_rf):.3f}')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=0.8, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Defection Prediction: ROC Curves', fontweight='bold')
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig4_roc.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig4_roc.png"))
    plt.close()
    print("Figure 4 saved")


def figure5_sensitivity():
    thresholds = sorted([int(k) for k in sensitivity_results.keys()])
    aucs = [sensitivity_results[str(t)]['auc'] for t in thresholds]
    f1s = [sensitivity_results[str(t)]['f1'] for t in thresholds]
    pcts = [sensitivity_results[str(t)]['pct_defectors_test'] * 100 for t in thresholds]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    bar_width = 1.5
    x = np.array(thresholds)

    bars = ax1.bar(x - bar_width/2, aucs, bar_width, color='#4292c6', alpha=0.8, label='AUC')
    bars2 = ax1.bar(x + bar_width/2, f1s, bar_width, color='#ef6548', alpha=0.8, label='F1 Score')

    ax1.set_xlabel('Defection Threshold (%)')
    ax1.set_ylabel('Score')
    ax1.set_title('Defection Prediction Across Thresholds', fontweight='bold')
    ax1.set_xticks(thresholds)
    ax1.set_xticklabels([f'{t}%' for t in thresholds])
    ax1.set_ylim(0, 1.1)

    ax2 = ax1.twinx()
    ax2.plot(thresholds, pcts, 'ko-', markersize=8, linewidth=2, label='% Defectors (test)')
    ax2.set_ylabel('Percentage Classified as Defectors')
    ax2.set_ylim(0, max(pcts) * 1.3)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', frameon=True)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig5_sensitivity.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig5_sensitivity.png"))
    plt.close()
    print("Figure 5 saved")


def figure6_causal():
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    event_order = ['tea_party', 'trump', 'post_trump']
    titles = ['Tea Party Wave (2010)', 'Trump Era (2016)', 'Post-Trump (2020)']
    metrics = ['fiedler', 'distance', 'overlap']
    metric_labels = ['Fiedler Value', 'Party Distance', 'Ideological Overlap']

    colors_before = '#4292c6'
    colors_after = '#ef6548'

    for idx, (event, title) in enumerate(zip(event_order, titles)):
        ax = axes[idx]
        if event not in causal_results:
            ax.set_visible(False)
            continue

        r = causal_results[event]
        before_vals = [r['fiedler_before'], r['distance_before']]
        after_vals = [r['fiedler_after'], r['distance_after']]
        diffs = [r['fiedler_diff'], r['distance_diff']]
        labels = ['Fiedler\nValue', 'Party\nDistance']

        x = np.arange(len(labels))
        width = 0.3

        bars1 = ax.bar(x - width/2, before_vals, width, color=colors_before, alpha=0.8, label='Before')
        bars2 = ax.bar(x + width/2, after_vals, width, color=colors_after, alpha=0.8, label='After')

        for i, diff in enumerate(diffs):
            sign = '+' if diff >= 0 else ''
            ax.annotate(f'{sign}{diff:.3f}', xy=(x[i] + width/2, after_vals[i]),
                       xytext=(0, 5), textcoords='offset points',
                       ha='center', fontsize=8, fontweight='bold',
                       color='#d62728' if abs(diff) > 0.05 else '#555555')

        ax.set_title(title, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend(frameon=True, fontsize=8)
        ax.set_ylabel('Value')

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig6_causal.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig6_causal.png"))
    plt.close()
    print("Figure 6 saved")


def figure7_embeddings():
    emb_path = os.path.join(MODEL_DIR, "embeddings.npz")
    if not os.path.exists(emb_path):
        print("Skipping Figure 7: no embeddings")
        return

    emb_data = np.load(emb_path)
    target_congresses = [104, 112, 117]

    fig, axes = plt.subplots(1, len(target_congresses), figsize=(14, 4.5))

    # First pass: compute t-SNE for all panels and collect coordinate ranges
    all_coords = []
    all_parties = []
    panel_valid = []
    for idx, congress in enumerate(target_congresses):
        ckey = str(congress)
        if ckey not in emb_data.files:
            all_coords.append(None)
            all_parties.append(None)
            panel_valid.append(False)
            continue

        embeddings = emb_data[ckey]
        with open(os.path.join(RESULTS_DIR, f"member_info_{congress}.json")) as f:
            mi = json.load(f)
        data = np.load(os.path.join(RESULTS_DIR, f"congress_{congress}.npz"))
        ml = data['member_list']

        parties = np.array([mi.get(str(int(m)), {}).get('party', 0) for m in ml])

        perplexity = min(30, len(ml) - 1)
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        coords = tsne.fit_transform(embeddings)

        all_coords.append(coords)
        all_parties.append(parties)
        panel_valid.append(True)

    # Compute shared axis limits across all valid panels
    valid_coords = [c for c in all_coords if c is not None]
    if valid_coords:
        all_x = np.concatenate([c[:, 0] for c in valid_coords])
        all_y = np.concatenate([c[:, 1] for c in valid_coords])
        x_margin = (all_x.max() - all_x.min()) * 0.08
        y_margin = (all_y.max() - all_y.min()) * 0.08
        shared_xlim = (all_x.min() - x_margin, all_x.max() + x_margin)
        shared_ylim = (all_y.min() - y_margin, all_y.max() + y_margin)

    # Second pass: plot with shared limits
    for idx, congress in enumerate(target_congresses):
        ax = axes[idx]
        if not panel_valid[idx]:
            ax.set_visible(False)
            continue

        coords = all_coords[idx]
        parties = all_parties[idx]
        dem_mask = parties == 100
        rep_mask = parties == 200

        ax.scatter(coords[dem_mask, 0], coords[dem_mask, 1], c=DEM_COLOR, s=15, alpha=0.6, label='Democrat')
        ax.scatter(coords[rep_mask, 0], coords[rep_mask, 1], c=REP_COLOR, s=15, alpha=0.6, label='Republican')

        year = CONGRESS_YEARS.get(congress, congress)
        ax.set_title(f'{congress}th Congress ({year})', fontweight='bold')
        ax.set_xlabel('t-SNE 1')
        ax.set_ylabel('t-SNE 2')
        ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(shared_xlim)
        ax.set_ylim(shared_ylim)

    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, "fig7_embeddings.pdf"))
    plt.savefig(os.path.join(FIG_DIR, "fig7_embeddings.png"))
    plt.close()
    print("Figure 7 saved")


if __name__ == "__main__":
    print("Generating publication figures...")
    figure1_network()
    figure2_polarization()
    figure3_attention()
    figure4_roc()
    figure5_sensitivity()
    figure6_causal()
    figure7_embeddings()
    print("\nAll figures saved to paper_figures/")
