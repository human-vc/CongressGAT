#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from sklearn.metrics import roc_auc_score, f1_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline_results")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model_results")
os.makedirs(MODEL_DIR, exist_ok=True)


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

        return out, attn


class TemporalAttention(nn.Module):
    def __init__(self, embed_dim, n_heads=2):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x_seq):
        attn_out, attn_weights = self.attention(x_seq, x_seq, x_seq)
        return self.norm(x_seq + attn_out), attn_weights


class CongressGAT(nn.Module):
    def __init__(self, in_features, hidden_dim=32, n_heads=4, n_temporal_heads=2, dropout=0.1):
        super().__init__()
        self.gat1 = GATLayer(in_features, hidden_dim, n_heads, dropout, concat=True)
        self.gat2 = GATLayer(hidden_dim * n_heads, hidden_dim, n_heads, dropout, concat=False)

        self.temporal_attention = TemporalAttention(hidden_dim, n_temporal_heads)

        self.polarization_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )

        self.defection_head = nn.Sequential(
            nn.Linear(hidden_dim + in_features, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.coalition_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

        self.dropout = nn.Dropout(dropout)

    def encode_graph(self, x, adj):
        h, attn1 = self.gat1(x, adj)
        h = F.elu(h)
        h = self.dropout(h)
        h, attn2 = self.gat2(h, adj)
        h = F.elu(h)
        return h, attn1, attn2

    def forward_temporal(self, embeddings_seq):
        seq = torch.stack(embeddings_seq, dim=0).unsqueeze(0)
        out, temporal_attn = self.temporal_attention(seq)
        return out.squeeze(0), temporal_attn

    def predict_polarization(self, graph_embedding):
        return self.polarization_head(graph_embedding)

    def predict_defection(self, node_embedding, node_features):
        combined = torch.cat([node_embedding, node_features], dim=-1)
        return self.defection_head(combined)

    def predict_coalition(self, node_i, node_j):
        combined = torch.cat([node_i, node_j], dim=-1)
        return self.coalition_head(combined)


def load_congress_data(congress):
    npz_path = os.path.join(RESULTS_DIR, f"congress_{congress}.npz")
    if not os.path.exists(npz_path):
        return None

    data = np.load(npz_path, allow_pickle=True)
    info_path = os.path.join(RESULTS_DIR, f"member_info_{congress}.json")
    with open(info_path) as f:
        member_info = json.load(f)

    def_rates = np.load(os.path.join(RESULTS_DIR, f"defection_rates_{congress}.npy"))
    def_labels = np.load(os.path.join(RESULTS_DIR, f"defection_labels_{congress}.npy"))

    return {
        'agreement': data['agreement'],
        'vote_matrix': data['vote_matrix'],
        'features': data['node_features'],
        'member_list': data['member_list'],
        'member_info': member_info,
        'defection_rates': def_rates,
        'defection_labels': def_labels,
    }


def train_model(train_congresses, test_congresses, device='cpu'):
    print("Loading training data...")
    train_data = {}
    for c in train_congresses:
        d = load_congress_data(c)
        if d is not None:
            train_data[c] = d
            print(f"  Congress {c}: {len(d['member_list'])} members")

    test_data = {}
    for c in test_congresses:
        d = load_congress_data(c)
        if d is not None:
            test_data[c] = d
            print(f"  Test Congress {c}: {len(d['member_list'])} members")

    with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
        all_results = json.load(f)

    in_features = 8
    model = CongressGAT(in_features=in_features, hidden_dim=32, n_heads=4, dropout=0.1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    all_attention_weights = {}

    print("\nTraining...")
    for epoch in range(200):
        model.train()
        total_loss = 0
        n_batches = 0

        graph_embeddings = []
        polarization_targets = []
        congress_list_ordered = sorted(train_data.keys())

        for congress in congress_list_ordered:
            d = train_data[congress]
            x = torch.FloatTensor(d['features']).to(device)
            adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float)).to(device)

            node_emb, attn1, attn2 = model.encode_graph(x, adj)
            graph_emb = node_emb.mean(dim=0)
            graph_embeddings.append(graph_emb)

            ckey = str(congress)
            if ckey in all_results:
                target = all_results[ckey]['spectral']['fiedler']
                polarization_targets.append(target)

            def_labels = torch.FloatTensor(d['defection_labels']).to(device)
            def_pred = model.predict_defection(node_emb, x).squeeze()
            def_loss = F.binary_cross_entropy(def_pred, def_labels)

            parties = []
            for icpsr in d['member_list']:
                info = d['member_info'].get(str(int(icpsr)), {})
                parties.append(1 if info.get('party') == 200 else 0)
            parties = np.array(parties)

            n_pairs = min(200, len(d['member_list']) * (len(d['member_list']) - 1) // 2)
            coalition_loss = torch.tensor(0.0, device=device)
            if n_pairs > 0:
                idx_pairs = []
                for _ in range(n_pairs):
                    i = np.random.randint(len(d['member_list']))
                    j = np.random.randint(len(d['member_list']))
                    if i != j:
                        idx_pairs.append((i, j))

                if idx_pairs:
                    ii = [p[0] for p in idx_pairs]
                    jj = [p[1] for p in idx_pairs]
                    same_party = torch.FloatTensor([
                        1.0 if parties[i] == parties[j] else 0.0
                        for i, j in idx_pairs
                    ]).to(device)
                    coal_pred = model.predict_coalition(
                        node_emb[ii], node_emb[jj]
                    ).squeeze()
                    coalition_loss = F.binary_cross_entropy(coal_pred, same_party)

            loss = def_loss + 0.5 * coalition_loss
            total_loss += loss.item()
            n_batches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if polarization_targets:
            model.train()
            optimizer.zero_grad()

            re_embeddings = []
            for congress in congress_list_ordered:
                d = train_data[congress]
                x = torch.FloatTensor(d['features']).to(device)
                adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float)).to(device)
                node_emb, _, _ = model.encode_graph(x, adj)
                re_embeddings.append(node_emb.mean(dim=0))

            targets = torch.FloatTensor(polarization_targets).to(device)
            temporal_out, temp_attn = model.forward_temporal(re_embeddings)
            pol_pred = model.predict_polarization(temporal_out).squeeze()

            if pol_pred.dim() == 0:
                pol_pred = pol_pred.unsqueeze(0)
            if targets.dim() == 0:
                targets = targets.unsqueeze(0)

            min_len = min(len(pol_pred), len(targets))
            pol_loss = F.mse_loss(pol_pred[:min_len], targets[:min_len])

            pol_loss.backward()
            optimizer.step()

            total_loss += pol_loss.item()

        scheduler.step()

        if (epoch + 1) % 50 == 0:
            print(f"  Epoch {epoch+1}: loss={total_loss/max(n_batches,1):.4f}")

    print("\nEvaluating...")
    model.eval()
    results = {
        'polarization': {},
        'defection': {},
        'coalition': {},
        'attention': {},
    }

    with torch.no_grad():
        all_embeddings = {}
        all_attentions = {}

        all_congress_keys = sorted(set(list(train_data.keys()) + list(test_data.keys())))
        all_available = {}
        for c in all_congress_keys:
            if c in train_data:
                all_available[c] = train_data[c]
            elif c in test_data:
                all_available[c] = test_data[c]

        graph_embs_all = []
        pol_targets_all = []
        pol_splits = []

        for congress in sorted(all_available.keys()):
            d = all_available[congress]
            x = torch.FloatTensor(d['features']).to(device)
            adj = torch.FloatTensor((d['agreement'] > 0.5).astype(float)).to(device)

            node_emb, attn1, attn2 = model.encode_graph(x, adj)
            graph_emb = node_emb.mean(dim=0)
            graph_embs_all.append(graph_emb)

            all_embeddings[congress] = node_emb.cpu().numpy()
            all_attentions[congress] = {
                'layer1': attn1.cpu().numpy(),
                'layer2': attn2.cpu().numpy(),
            }

            ckey = str(congress)
            if ckey in all_results:
                pol_targets_all.append(all_results[ckey]['spectral']['fiedler'])
                pol_splits.append('test' if congress in test_data else 'train')

            def_labels = d['defection_labels']
            def_pred = model.predict_defection(node_emb, x).squeeze().cpu().numpy()

            try:
                auc = roc_auc_score(def_labels, def_pred)
            except:
                auc = 0.5
            f1 = f1_score(def_labels, (def_pred > 0.5).astype(int), zero_division=0)

            results['defection'][congress] = {
                'auc': float(auc),
                'f1': float(f1),
                'n_defectors': int(def_labels.sum()),
                'n_total': int(len(def_labels)),
                'predictions': def_pred.tolist(),
                'labels': def_labels.tolist(),
                'split': 'test' if congress in test_data else 'train',
            }

            parties = []
            for icpsr in d['member_list']:
                info = d['member_info'].get(str(int(icpsr)), {})
                parties.append(1 if info.get('party') == 200 else 0)
            parties = np.array(parties)

            n_eval = min(500, len(d['member_list']) * 2)
            coal_true = []
            coal_pred_list = []
            for _ in range(n_eval):
                i = np.random.randint(len(d['member_list']))
                j = np.random.randint(len(d['member_list']))
                if i != j:
                    pred = model.predict_coalition(
                        node_emb[i:i+1], node_emb[j:j+1]
                    ).item()
                    coal_pred_list.append(pred)
                    coal_true.append(1.0 if parties[i] == parties[j] else 0.0)

            if coal_true:
                coal_true = np.array(coal_true)
                coal_pred_arr = np.array(coal_pred_list)
                try:
                    coal_auc = roc_auc_score(coal_true, coal_pred_arr)
                except:
                    coal_auc = 0.5
                coal_f1 = f1_score(coal_true, (coal_pred_arr > 0.5).astype(int), zero_division=0)
                results['coalition'][congress] = {
                    'auc': float(coal_auc),
                    'f1': float(coal_f1),
                    'split': 'test' if congress in test_data else 'train',
                }

        if graph_embs_all:
            temporal_out, temp_attn = model.forward_temporal(graph_embs_all)
            pol_pred = model.predict_polarization(temporal_out).squeeze().cpu().numpy()
            if pol_pred.ndim == 0:
                pol_pred = np.array([pol_pred])

            min_len = min(len(pol_pred), len(pol_targets_all))
            for i in range(min_len):
                c = sorted(all_available.keys())[i]
                results['polarization'][c] = {
                    'predicted': float(pol_pred[i]),
                    'actual': float(pol_targets_all[i]),
                    'split': pol_splits[i],
                }

            results['attention']['temporal'] = temp_attn.cpu().numpy().tolist()

    np.savez_compressed(
        os.path.join(MODEL_DIR, "embeddings.npz"),
        **{str(k): v for k, v in all_embeddings.items()}
    )

    for congress, attn_data in all_attentions.items():
        np.savez_compressed(
            os.path.join(MODEL_DIR, f"attention_{congress}.npz"),
            layer1=attn_data['layer1'],
            layer2=attn_data['layer2'],
        )

    torch.save(model.state_dict(), os.path.join(MODEL_DIR, "model.pt"))

    print("\n--- RESULTS ---")
    print("\nDefection Prediction (AUC):")
    train_aucs = []
    test_aucs = []
    for c in sorted(results['defection'].keys()):
        r = results['defection'][c]
        label = "TEST" if r['split'] == 'test' else "train"
        print(f"  Congress {c} [{label}]: AUC={r['auc']:.3f}, F1={r['f1']:.3f}")
        if r['split'] == 'test':
            test_aucs.append(r['auc'])
        else:
            train_aucs.append(r['auc'])

    print(f"\n  Train mean AUC: {np.mean(train_aucs):.3f}")
    if test_aucs:
        print(f"  Test mean AUC: {np.mean(test_aucs):.3f}")

    print("\nCoalition Detection (F1):")
    for c in sorted(results['coalition'].keys()):
        r = results['coalition'][c]
        label = "TEST" if r['split'] == 'test' else "train"
        print(f"  Congress {c} [{label}]: AUC={r['auc']:.3f}, F1={r['f1']:.3f}")

    print("\nPolarization Prediction:")
    for c in sorted(results['polarization'].keys()):
        r = results['polarization'][c]
        label = "TEST" if r['split'] == 'test' else "train"
        print(f"  Congress {c} [{label}]: pred={r['predicted']:.4f}, actual={r['actual']:.4f}")

    with open(os.path.join(MODEL_DIR, "results.json"), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    return model, results


def run_baselines(train_congresses, test_congresses):
    print("\n\n=== BASELINES ===")
    baseline_results = {}

    train_X, train_y = [], []
    test_X, test_y = [], []
    test_congress_labels = []

    for congress in train_congresses:
        d = load_congress_data(congress)
        if d is None:
            continue
        train_X.append(d['features'])
        train_y.append(d['defection_labels'])

    for congress in test_congresses:
        d = load_congress_data(congress)
        if d is None:
            continue
        test_X.append(d['features'])
        test_y.append(d['defection_labels'])
        test_congress_labels.extend([congress] * len(d['defection_labels']))

    train_X = np.vstack(train_X) if train_X else np.array([])
    train_y = np.concatenate(train_y) if train_y else np.array([])
    test_X = np.vstack(test_X) if test_X else np.array([])
    test_y = np.concatenate(test_y) if test_y else np.array([])

    print(f"Train: {len(train_y)} samples, Test: {len(test_y)} samples")
    print(f"Train defection rate: {train_y.mean():.3f}")
    print(f"Test defection rate: {test_y.mean():.3f}")

    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(train_X, train_y)
    lr_pred = lr.predict_proba(test_X)[:, 1]
    lr_auc = roc_auc_score(test_y, lr_pred)
    lr_f1 = f1_score(test_y, (lr_pred > 0.5).astype(int), zero_division=0)
    print(f"\nLogistic Regression: AUC={lr_auc:.3f}, F1={lr_f1:.3f}")
    baseline_results['logistic_regression'] = {'auc': lr_auc, 'f1': lr_f1}

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(train_X, train_y)
    rf_pred = rf.predict_proba(test_X)[:, 1]
    rf_auc = roc_auc_score(test_y, rf_pred)
    rf_f1 = f1_score(test_y, (rf_pred > 0.5).astype(int), zero_division=0)
    print(f"Random Forest: AUC={rf_auc:.3f}, F1={rf_f1:.3f}")
    baseline_results['random_forest'] = {'auc': rf_auc, 'f1': rf_f1}

    with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
        all_results = json.load(f)

    fiedler_series = []
    for c in sorted(all_results.keys(), key=int):
        fiedler_series.append({
            'congress': int(c),
            'fiedler': all_results[c]['spectral']['fiedler'],
        })

    if len(fiedler_series) >= 2:
        naive_preds = []
        naive_actual = []
        for i in range(1, len(fiedler_series)):
            naive_preds.append(fiedler_series[i-1]['fiedler'])
            naive_actual.append(fiedler_series[i]['fiedler'])
        naive_mse = mean_squared_error(naive_actual, naive_preds)
        naive_mae = mean_absolute_error(naive_actual, naive_preds)
        print(f"\nNaive Drift (polarization): MSE={naive_mse:.6f}, MAE={naive_mae:.4f}")
        baseline_results['naive_drift'] = {'mse': naive_mse, 'mae': naive_mae}

    baseline_results['predictions'] = {
        'lr': lr_pred.tolist(),
        'rf': rf_pred.tolist(),
        'test_labels': test_y.tolist(),
        'test_congress': test_congress_labels,
    }

    with open(os.path.join(MODEL_DIR, "baseline_results.json"), 'w') as f:
        json.dump(baseline_results, f, indent=2, default=str)

    return baseline_results


def run_defection_sensitivity(train_congresses, test_congresses):
    print("\n\n=== DEFECTION THRESHOLD SENSITIVITY ===")
    sensitivity = {}

    for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
        train_X, train_y = [], []
        test_X, test_y = [], []

        for congress in train_congresses:
            d = load_congress_data(congress)
            if d is None:
                continue
            _, labels = compute_defection_from_stored(congress, thresh)
            if labels is not None:
                train_X.append(d['features'])
                train_y.append(labels)

        for congress in test_congresses:
            d = load_congress_data(congress)
            if d is None:
                continue
            _, labels = compute_defection_from_stored(congress, thresh)
            if labels is not None:
                test_X.append(d['features'])
                test_y.append(labels)

        if not train_X or not test_X:
            continue

        train_X = np.vstack(train_X)
        train_y = np.concatenate(train_y)
        test_X = np.vstack(test_X)
        test_y = np.concatenate(test_y)

        if test_y.sum() == 0 or test_y.sum() == len(test_y):
            continue

        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(train_X, train_y)
        rf_pred = rf.predict_proba(test_X)[:, 1]
        try:
            auc = roc_auc_score(test_y, rf_pred)
        except:
            auc = 0.5
        f1 = f1_score(test_y, (rf_pred > 0.5).astype(int), zero_division=0)

        pct = int(thresh * 100)
        sensitivity[pct] = {
            'auc': float(auc),
            'f1': float(f1),
            'n_defectors_train': int(train_y.sum()),
            'n_defectors_test': int(test_y.sum()),
            'pct_defectors_test': float(test_y.mean()),
        }
        print(f"  Threshold {pct}%: AUC={auc:.3f}, F1={f1:.3f}, defectors={test_y.sum()}/{len(test_y)}")

    with open(os.path.join(MODEL_DIR, "sensitivity_results.json"), 'w') as f:
        json.dump(sensitivity, f, indent=2)

    return sensitivity


def compute_defection_from_stored(congress, threshold):
    rates_path = os.path.join(RESULTS_DIR, f"defection_rates_{congress}.npy")
    if not os.path.exists(rates_path):
        return None, None
    rates = np.load(rates_path)
    labels = (rates >= threshold).astype(int)
    return rates, labels


def run_causal_analysis():
    print("\n\n=== CAUSAL ANALYSIS (Diff-in-Diff) ===")

    with open(os.path.join(RESULTS_DIR, "all_results.json")) as f:
        all_results = json.load(f)

    causal_results = {}

    events = {
        'tea_party': {'before': [110, 111], 'after': [112, 113], 'label': 'Tea Party Wave (2010)'},
        'trump': {'before': [113, 114], 'after': [115, 116], 'label': 'Trump Era (2016)'},
        'post_trump': {'before': [115, 116], 'after': [117, 118], 'label': 'Post-Trump (2020)'},
    }

    for event_name, event_info in events.items():
        print(f"\n  {event_info['label']}:")

        before_fiedler = []
        after_fiedler = []
        before_distance = []
        after_distance = []
        before_overlap = []
        after_overlap = []

        for c in event_info['before']:
            ckey = str(c)
            if ckey in all_results:
                before_fiedler.append(all_results[ckey]['spectral']['fiedler'])
                before_distance.append(all_results[ckey]['polarization']['party_distance'])
                before_overlap.append(all_results[ckey]['polarization']['overlap'])

        for c in event_info['after']:
            ckey = str(c)
            if ckey in all_results:
                after_fiedler.append(all_results[ckey]['spectral']['fiedler'])
                after_distance.append(all_results[ckey]['polarization']['party_distance'])
                after_overlap.append(all_results[ckey]['polarization']['overlap'])

        if before_fiedler and after_fiedler:
            diff_fiedler = np.mean(after_fiedler) - np.mean(before_fiedler)
            diff_distance = np.mean(after_distance) - np.mean(before_distance)
            diff_overlap = np.mean(after_overlap) - np.mean(before_overlap)

            causal_results[event_name] = {
                'label': event_info['label'],
                'before_congresses': event_info['before'],
                'after_congresses': event_info['after'],
                'fiedler_before': float(np.mean(before_fiedler)),
                'fiedler_after': float(np.mean(after_fiedler)),
                'fiedler_diff': float(diff_fiedler),
                'distance_before': float(np.mean(before_distance)),
                'distance_after': float(np.mean(after_distance)),
                'distance_diff': float(diff_distance),
                'overlap_before': float(np.mean(before_overlap)),
                'overlap_after': float(np.mean(after_overlap)),
                'overlap_diff': float(diff_overlap),
            }

            print(f"    Fiedler: {np.mean(before_fiedler):.4f} -> {np.mean(after_fiedler):.4f} (diff={diff_fiedler:+.4f})")
            print(f"    Party Distance: {np.mean(before_distance):.4f} -> {np.mean(after_distance):.4f} (diff={diff_distance:+.4f})")
            print(f"    Overlap: {np.mean(before_overlap):.4f} -> {np.mean(after_overlap):.4f} (diff={diff_overlap:+.4f})")

    with open(os.path.join(MODEL_DIR, "causal_results.json"), 'w') as f:
        json.dump(causal_results, f, indent=2)

    return causal_results


if __name__ == "__main__":
    train_congresses = list(range(104, 115))
    test_congresses = [115, 116, 117]

    model, results = train_model(train_congresses, test_congresses)
    baseline_results = run_baselines(train_congresses, test_congresses)
    sensitivity = run_defection_sensitivity(train_congresses, test_congresses)
    causal = run_causal_analysis()

    print("\n\nDONE. All results saved to model_results/")
