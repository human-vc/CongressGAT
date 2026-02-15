#!/usr/bin/env python3
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline_results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_results")

class GATLayer(nn.Module):
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.1, concat=True):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.concat = concat
        self.W = nn.Parameter(torch.FloatTensor(n_heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.FloatTensor(n_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.FloatTensor(n_heads, out_features, 1))
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(dropout)
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, x, adj):
        n = x.size(0)
        h = torch.einsum('ni,hio->hno', x, self.W)
        attn_src = torch.einsum('hno,hod->hnd', h, self.a_src).squeeze(-1)
        attn_dst = torch.einsum('hno,hod->hnd', h, self.a_dst).squeeze(-1)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leaky_relu(attn)
        mask = (adj == 0).unsqueeze(0).expand(self.n_heads, -1, -1)
        attn = attn.masked_fill(mask, float('-inf'))
        attn = F.softmax(attn, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)
        attn = self.dropout(attn)
        out = torch.einsum('hnm,hmo->hno', attn, h)
        if self.concat:
            out = out.permute(1, 0, 2).contiguous().view(n, -1)
        else:
            out = out.mean(dim=0)
        return out

class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_seq):
        attn_out, _ = self.attention(x_seq, x_seq, x_seq)
        return self.norm(x_seq + attn_out)

class CongressGATAblation(nn.Module):
    def __init__(self, in_features, hidden_dim=32, n_heads=4, dropout=0.1):
        super().__init__()
        self.in_features = in_features
        self.gat1 = GATLayer(in_features, hidden_dim, n_heads, dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout, concat=False)
        self.temporal_attention = TemporalAttention(hidden_dim, 2)
        self.polarization_head = nn.Sequential(
            nn.Linear(hidden_dim, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1))
        self.defection_head = nn.Sequential(
            nn.Linear(hidden_dim + in_features, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid())
        self.coalition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(32, 1), nn.Sigmoid())
        self.dropout = nn.Dropout(dropout)

    def encode_graph(self, x, adj):
        h = self.gat1(x, adj)
        h = F.elu(h)
        h = self.dropout(h)
        h = self.gat2(h, adj)
        h = F.elu(h)
        return h

    def forward_temporal(self, embeddings_seq):
        seq = torch.stack(embeddings_seq, dim=0).unsqueeze(0)
        out = self.temporal_attention(seq)
        return out.squeeze(0)


def load_congress_data(congress):
    npz_path = os.path.join(RESULTS_DIR, f"congress_{congress}.npz")
    if not os.path.exists(npz_path):
        return None
    data = np.load(npz_path, allow_pickle=True)
    info_path = os.path.join(RESULTS_DIR, f"member_info_{congress}.json")
    with open(info_path) as f:
        member_info = json.load(f)
    def_labels = np.load(os.path.join(RESULTS_DIR, f"defection_labels_{congress}.npy"))
    return {
        'features': data['node_features'],
        'agreement': data['agreement'],
        'member_list': data['member_list'],
        'member_info': member_info,
        'defection_labels': def_labels,
    }


def train_and_eval(feature_mask, label, train_congresses, test_congresses):
    torch.manual_seed(42)
    np.random.seed(42)

    with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
        all_results = json.load(f)

    train_data = {}
    test_data = {}
    for c in train_congresses:
        d = load_congress_data(c)
        if d: train_data[c] = d
    for c in test_congresses:
        d = load_congress_data(c)
        if d: test_data[c] = d

    in_features = sum(feature_mask)
    model = CongressGATAblation(in_features=in_features, hidden_dim=32, n_heads=4, dropout=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    feat_idx = [i for i, m in enumerate(feature_mask) if m]

    for epoch in range(200):
        model.train()
        congress_list = sorted(train_data.keys())

        for congress in congress_list:
            d = train_data[congress]
            x = torch.FloatTensor(d['features'][:, feat_idx])
            adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float))

            node_emb = model.encode_graph(x, adj)

            def_labels = torch.FloatTensor(d['defection_labels'])
            combined = torch.cat([node_emb, x], dim=-1)
            def_pred = model.defection_head(combined).squeeze()
            def_loss = F.binary_cross_entropy(def_pred, def_labels)

            parties = np.array([
                1 if d['member_info'].get(str(int(ic)), {}).get('party') == 200 else 0
                for ic in d['member_list']
            ])

            idx_pairs = []
            for _ in range(200):
                i, j = np.random.randint(len(d['member_list'])), np.random.randint(len(d['member_list']))
                if i != j: idx_pairs.append((i, j))

            if idx_pairs:
                ii = [p[0] for p in idx_pairs]
                jj = [p[1] for p in idx_pairs]
                same_party = torch.FloatTensor([1.0 if parties[i]==parties[j] else 0.0 for i,j in idx_pairs])
                coal_pred = model.coalition_head(torch.cat([node_emb[ii], node_emb[jj]], dim=-1)).squeeze()
                coal_loss = F.binary_cross_entropy(coal_pred, same_party)
            else:
                coal_loss = torch.tensor(0.0)

            loss = def_loss + 0.5 * coal_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        model.train()
        optimizer.zero_grad()
        graph_embs = []
        pol_targets = []
        for congress in congress_list:
            d = train_data[congress]
            x = torch.FloatTensor(d['features'][:, feat_idx])
            adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float))
            node_emb = model.encode_graph(x, adj)
            graph_embs.append(node_emb.mean(dim=0))
            ckey = str(congress)
            if ckey in all_results:
                pol_targets.append(all_results[ckey]['spectral']['fiedler'])

        if pol_targets:
            temporal_out = model.forward_temporal(graph_embs)
            pol_pred = model.polarization_head(temporal_out).squeeze()
            if pol_pred.dim() == 0: pol_pred = pol_pred.unsqueeze(0)
            targets = torch.FloatTensor(pol_targets)
            min_len = min(len(pol_pred), len(targets))
            pol_loss = F.mse_loss(pol_pred[:min_len], targets[:min_len])
            pol_loss.backward()
            optimizer.step()

        scheduler.step()

    model.eval()
    results_out = {'defection': {}, 'coalition': {}, 'polarization': {}}

    with torch.no_grad():
        all_available = {}
        for c in sorted(set(list(train_data.keys()) + list(test_data.keys()))):
            all_available[c] = train_data.get(c) or test_data.get(c)

        graph_embs_all = []
        pol_targets_all = []

        for congress in sorted(all_available.keys()):
            d = all_available[congress]
            x = torch.FloatTensor(d['features'][:, feat_idx])
            adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float))
            node_emb = model.encode_graph(x, adj)
            graph_embs_all.append(node_emb.mean(dim=0))

            ckey = str(congress)
            if ckey in all_results:
                pol_targets_all.append(all_results[ckey]['spectral']['fiedler'])

            def_labels = d['defection_labels']
            combined = torch.cat([node_emb, x], dim=-1)
            def_pred = model.defection_head(combined).squeeze().cpu().numpy()
            try: auc = roc_auc_score(def_labels, def_pred)
            except: auc = 0.5
            f1 = f1_score(def_labels, (def_pred > 0.5).astype(int), zero_division=0)
            split = 'test' if congress in test_data else 'train'
            results_out['defection'][congress] = {'auc': float(auc), 'f1': float(f1), 'split': split}

            parties = np.array([
                1 if d['member_info'].get(str(int(ic)), {}).get('party') == 200 else 0
                for ic in d['member_list']
            ])
            coal_true, coal_preds = [], []
            for _ in range(500):
                i, j = np.random.randint(len(d['member_list'])), np.random.randint(len(d['member_list']))
                if i != j:
                    pred = model.coalition_head(torch.cat([node_emb[i:i+1], node_emb[j:j+1]], dim=-1)).item()
                    coal_preds.append(pred)
                    coal_true.append(1.0 if parties[i]==parties[j] else 0.0)

            if coal_true:
                coal_true_arr = np.array(coal_true)
                coal_preds_arr = np.array(coal_preds)
                try: coal_auc = roc_auc_score(coal_true_arr, coal_preds_arr)
                except: coal_auc = 0.5
                coal_f1 = f1_score(coal_true_arr, (coal_preds_arr > 0.5).astype(int), zero_division=0)
                results_out['coalition'][congress] = {'auc': float(coal_auc), 'f1': float(coal_f1), 'split': split}

        if graph_embs_all and pol_targets_all:
            temporal_out = model.forward_temporal(graph_embs_all)
            pol_pred = model.polarization_head(temporal_out).squeeze().cpu().numpy()
            if pol_pred.ndim == 0: pol_pred = np.array([pol_pred])
            min_len = min(len(pol_pred), len(pol_targets_all))
            for i in range(min_len):
                c = sorted(all_available.keys())[i]
                split = 'test' if c in test_data else 'train'
                results_out['polarization'][c] = {
                    'predicted': float(pol_pred[i]),
                    'actual': float(pol_targets_all[i]),
                    'split': split
                }

    test_def_aucs = [v['auc'] for v in results_out['defection'].values() if v['split'] == 'test']
    test_def_f1s = [v['f1'] for v in results_out['defection'].values() if v['split'] == 'test']
    test_coal_f1s = [v['f1'] for v in results_out['coalition'].values() if v['split'] == 'test']
    test_coal_aucs = [v['auc'] for v in results_out['coalition'].values() if v['split'] == 'test']

    test_pol = [(v['predicted'], v['actual']) for v in results_out['polarization'].values() if v['split'] == 'test']
    if test_pol:
        pol_mse = np.mean([(p - a)**2 for p, a in test_pol])
    else:
        pol_mse = float('nan')

    summary = {
        'label': label,
        'n_features': in_features,
        'feature_indices': feat_idx,
        'test_defection_auc': float(np.mean(test_def_aucs)) if test_def_aucs else 0,
        'test_defection_f1': float(np.mean(test_def_f1s)) if test_def_f1s else 0,
        'test_coalition_f1': float(np.mean(test_coal_f1s)) if test_coal_f1s else 0,
        'test_coalition_auc': float(np.mean(test_coal_aucs)) if test_coal_aucs else 0,
        'test_polarization_mse': float(pol_mse),
        'per_congress_defection': {str(k): v for k, v in results_out['defection'].items() if v['split'] == 'test'},
        'per_congress_coalition': {str(k): v for k, v in results_out['coalition'].items() if v['split'] == 'test'},
    }

    return summary


if __name__ == "__main__":
    train_congresses = list(range(104, 115))
    test_congresses = [115, 116, 117]

    ablations = {
        'Full (all 8 features)':      [True, True, True, True, True, True, True, True],
        'No DW-NOMINATE':             [False, False, True, True, True, True, True, True],
        'No Party ID':                [True, True, False, True, True, True, True, True],
        'No NOMINATE + No Party':     [False, False, False, True, True, True, True, True],
        'Network-only (agr features)':[False, False, False, False, False, True, True, True],
        'NOMINATE-only':              [True, True, False, False, False, False, False, False],
    }

    all_ablation_results = {}

    for label, mask in ablations.items():
        print(f"\n{'='*60}")
        print(f"ABLATION: {label}")
        print(f"Features: {[i for i,m in enumerate(mask) if m]} ({sum(mask)} total)")
        print(f"{'='*60}")

        result = train_and_eval(mask, label, train_congresses, test_congresses)
        all_ablation_results[label] = result

        print(f"  Defection AUC: {result['test_defection_auc']:.3f}")
        print(f"  Defection F1:  {result['test_defection_f1']:.3f}")
        print(f"  Coalition F1:  {result['test_coalition_f1']:.3f}")
        print(f"  Polarization MSE: {result['test_polarization_mse']:.6f}")

    with open(os.path.join(MODEL_DIR, "ablation_results.json"), 'w') as f:
        json.dump(all_ablation_results, f, indent=2)

    print("\n\n=== ABLATION SUMMARY ===")
    print(f"{'Configuration':<35} {'Def AUC':>8} {'Def F1':>8} {'Coal F1':>8} {'Pol MSE':>10}")
    print("-" * 75)
    for label, r in all_ablation_results.items():
        print(f"{label:<35} {r['test_defection_auc']:>8.3f} {r['test_defection_f1']:>8.3f} {r['test_coalition_f1']:>8.3f} {r['test_polarization_mse']:>10.6f}")

    print("\nResults saved to model_results/ablation_results.json")
