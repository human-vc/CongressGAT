import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv
from torch_geometric.data import Data
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Fixed seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

RESULTS_DIR = os.path.expanduser('~/CongressionalGNN/results_final')
CONGRESSES = list(range(104, 119))
TRAIN_CONGRESSES = list(range(104, 115))
TEST_CONGRESSES = [115, 116, 117]
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]

def load_processed_data():
    data = np.load(os.path.join(RESULTS_DIR, 'processed_data.npz'), allow_pickle=True)
    agreement_matrices, features, member_ids = {}, {}, {}
    for c in CONGRESSES:
        agreement_matrices[c] = data[f'agreement_{c}']
        features[c] = data[f'features_{c}']
        member_ids[c] = data[f'member_ids_{c}']
    defection_rates = {}
    for c in CONGRESSES:
        dr_data = np.load(os.path.join(RESULTS_DIR, f'defection_rates_{c}.npy'))
        defection_rates[c] = {int(row[0]): row[1] for row in dr_data}
    return agreement_matrices, features, member_ids, defection_rates

def build_knn_graph(agreement_matrix, k=20):
    n = agreement_matrix.shape[0]
    src_list, dst_list, weight_list = [], [], []
    for i in range(n):
        row = agreement_matrix[i].copy()
        row[i] = -1
        top_k = np.argsort(row)[-k:]
        for j in top_k:
            if row[j] > 0:
                src_list.append(i)
                dst_list.append(j)
                weight_list.append(row[j])
    edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_attr = torch.tensor(weight_list, dtype=torch.float32).unsqueeze(1)
    return edge_index, edge_attr

def build_pyg_graph(agreement_matrix, feats, def_rates, mids, threshold=0.10, k=20):
    n = len(mids)
    edge_index, edge_attr = build_knn_graph(agreement_matrix, k=k)
    labels = torch.tensor([1 if def_rates.get(mid, 0) > threshold else 0 for mid in mids], dtype=torch.long)
    x = torch.tensor(feats, dtype=torch.float32)
    rates = torch.tensor([def_rates.get(mid, 0) for mid in mids], dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=labels, defection_rates=rates, num_nodes=n)

class CongressGAT(nn.Module):
    def __init__(self, in_channels, hidden_channels=32, heads=4, dropout=0.3):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout, edge_dim=1)
        self.bn1 = nn.BatchNorm1d(hidden_channels * heads)
        self.conv2 = GATConv(hidden_channels * heads, hidden_channels, heads=1, dropout=dropout, concat=False, edge_dim=1)
        self.bn2 = nn.BatchNorm1d(hidden_channels)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 2)
        )
        self.regressor = nn.Sequential(
            nn.Linear(hidden_channels, 16), nn.ReLU(), nn.Dropout(dropout), nn.Linear(16, 1)
        )
        self.dropout = dropout

    def forward(self, data, return_attention=False):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x, attn1 = self.conv1(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x, attn2 = self.conv2(x, edge_index, edge_attr=edge_attr, return_attention_weights=True)
        x = self.bn2(x)
        x = F.elu(x)
        cls = self.classifier(x)
        reg = self.regressor(x).squeeze(-1)
        if return_attention:
            return cls, reg, x, (attn1, attn2)
        return cls, reg, x

def train_gat(graphs_train, graphs_test, threshold=0.10, epochs=200, lr=0.005):
    in_channels = graphs_train[0].x.shape[1]
    model = CongressGAT(in_channels, hidden_channels=32, heads=4, dropout=0.3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    pos_counts = sum(g.y.sum().item() for g in graphs_train)
    neg_counts = sum((g.y == 0).sum().item() for g in graphs_train)
    pos_weight = min(neg_counts / max(pos_counts, 1), 8.0)
    weight = torch.tensor([1.0, pos_weight], dtype=torch.float32)
    criterion_cls = nn.CrossEntropyLoss(weight=weight)
    criterion_reg = nn.SmoothL1Loss()

    best_val_f1 = -1
    best_state = None

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for g in graphs_train:
            optimizer.zero_grad()
            cls, reg, _ = model(g)
            loss = criterion_cls(cls, g.y) + 0.3 * criterion_reg(reg, g.defection_rates)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()

        if (epoch + 1) % 40 == 0:
            model.eval()
            with torch.no_grad():
                all_preds, all_labels = [], []
                for g in graphs_test:
                    cl, _, _ = model(g)
                    all_preds.extend(cl.argmax(dim=1).numpy())
                    all_labels.extend(g.y.numpy())
                f1 = f1_score(all_labels, all_preds, zero_division=0)
                acc = accuracy_score(all_labels, all_preds)
                sys.stdout.write(f"    Epoch {epoch+1}: loss={total_loss/len(graphs_train):.4f} acc={acc:.4f} f1={f1:.4f}\n")
                sys.stdout.flush()
                if f1 >= best_val_f1:
                    best_val_f1 = f1
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    return model

def evaluate_model(model, graphs):
    model.eval()
    all_preds, all_labels, all_probs = [], [], []
    all_rp, all_rl = [], []
    with torch.no_grad():
        for g in graphs:
            cl, rp, _ = model(g)
            probs = F.softmax(cl, dim=1)[:, 1].numpy()
            all_preds.extend(cl.argmax(dim=1).numpy())
            all_labels.extend(g.y.numpy())
            all_probs.extend(probs)
            all_rp.extend(rp.numpy())
            all_rl.extend(g.defection_rates.numpy())
    al, ap, aprobs = np.array(all_labels), np.array(all_preds), np.array(all_probs)
    m = {
        'accuracy': float(accuracy_score(al, ap)),
        'f1': float(f1_score(al, ap, zero_division=0)),
        'precision': float(precision_score(al, ap, zero_division=0)),
        'recall': float(recall_score(al, ap, zero_division=0)),
        'auc': float(roc_auc_score(al, aprobs)) if len(set(al)) > 1 else 0.5
    }
    rp, rl = np.array(all_rp), np.array(all_rl)
    m['reg_mse'] = float(np.mean((rp - rl) ** 2))
    m['reg_corr'] = float(np.corrcoef(rp, rl)[0, 1]) if len(rl) > 1 else 0
    return m

def run_baselines(graphs_train, graphs_test, threshold=0.10):
    X_train = np.vstack([g.x.numpy() for g in graphs_train])
    y_train = np.concatenate([g.y.numpy() for g in graphs_train])
    X_test = np.vstack([g.x.numpy() for g in graphs_test])
    y_test = np.concatenate([g.y.numpy() for g in graphs_test])
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)
    results = {}

    lr = LogisticRegression(max_iter=1000, class_weight='balanced')
    lr.fit(Xtr, y_train)
    lp = lr.predict(Xte)
    lpr = lr.predict_proba(Xte)[:, 1]
    results['logistic_regression'] = {
        'accuracy': float(accuracy_score(y_test, lp)),
        'f1': float(f1_score(y_test, lp, zero_division=0)),
        'precision': float(precision_score(y_test, lp, zero_division=0)),
        'recall': float(recall_score(y_test, lp, zero_division=0)),
        'auc': float(roc_auc_score(y_test, lpr)) if len(set(y_test)) > 1 else 0.5
    }

    rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=42, max_depth=8)
    rf.fit(Xtr, y_train)
    rp = rf.predict(Xte)
    rpr = rf.predict_proba(Xte)[:, 1]
    results['random_forest'] = {
        'accuracy': float(accuracy_score(y_test, rp)),
        'f1': float(f1_score(y_test, rp, zero_division=0)),
        'precision': float(precision_score(y_test, rp, zero_division=0)),
        'recall': float(recall_score(y_test, rp, zero_division=0)),
        'auc': float(roc_auc_score(y_test, rpr)) if len(set(y_test)) > 1 else 0.5
    }

    naive_preds = (X_test[:, 4] > threshold).astype(int)
    naive_probs = X_test[:, 4]
    results['naive_drift'] = {
        'accuracy': float(accuracy_score(y_test, naive_preds)),
        'f1': float(f1_score(y_test, naive_preds, zero_division=0)),
        'precision': float(precision_score(y_test, naive_preds, zero_division=0)),
        'recall': float(recall_score(y_test, naive_preds, zero_division=0)),
        'auc': float(roc_auc_score(y_test, naive_probs)) if len(set(y_test)) > 1 else 0.5
    }

    results['feature_importance'] = dict(zip(
        ['DW-NOM1', 'DW-NOM2', 'Party', 'Seniority', 'PrevDefection', 'Extremism'],
        rf.feature_importances_.tolist()
    ))
    return results

def run_full_experiment():
    print("Loading processed data...", flush=True)
    agreement_matrices, features, member_ids, defection_rates = load_processed_data()
    all_results = {}

    for threshold in THRESHOLDS:
        print(f"\n{'='*60}", flush=True)
        print(f"Threshold: {threshold*100:.0f}%", flush=True)
        print(f"{'='*60}", flush=True)

        graphs_train, graphs_test = [], []
        for c in TRAIN_CONGRESSES:
            g = build_pyg_graph(agreement_matrices[c], features[c], defection_rates[c], member_ids[c], threshold=threshold, k=15)
            graphs_train.append(g)
            print(f"  Train {c}: {g.num_nodes}n, {g.edge_index.shape[1]}e, {g.y.sum().item()} def", flush=True)
        for c in TEST_CONGRESSES:
            g = build_pyg_graph(agreement_matrices[c], features[c], defection_rates[c], member_ids[c], threshold=threshold, k=15)
            graphs_test.append(g)
            print(f"  Test  {c}: {g.num_nodes}n, {g.edge_index.shape[1]}e, {g.y.sum().item()} def", flush=True)

        print("  Training GAT...", flush=True)
        model = train_gat(graphs_train, graphs_test, threshold=threshold, epochs=200, lr=0.005)

        gat_train = evaluate_model(model, graphs_train)
        gat_test = evaluate_model(model, graphs_test)
        print(f"  GAT Train: Acc={gat_train['accuracy']:.4f} F1={gat_train['f1']:.4f} AUC={gat_train['auc']:.4f}", flush=True)
        print(f"  GAT Test:  Acc={gat_test['accuracy']:.4f} F1={gat_test['f1']:.4f} AUC={gat_test['auc']:.4f}", flush=True)

        print("  Baselines...", flush=True)
        baselines = run_baselines(graphs_train, graphs_test, threshold)
        for name, m in baselines.items():
            if name != 'feature_importance':
                print(f"  {name}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m.get('auc',0):.4f}", flush=True)

        all_results[f'threshold_{int(threshold*100)}'] = {
            'gat_train': gat_train, 'gat_test': gat_test, 'baselines': baselines
        }

    print("\nAttention analysis...", flush=True)
    graphs_all = []
    for c in CONGRESSES:
        g = build_pyg_graph(agreement_matrices[c], features[c], defection_rates[c], member_ids[c], threshold=0.10, k=15)
        graphs_all.append(g)

    model_final = train_gat(graphs_all[:11], graphs_all[11:], threshold=0.10, epochs=200, lr=0.005)

    attention_analysis = {}
    model_final.eval()
    with torch.no_grad():
        for c_idx, c in enumerate(CONGRESSES):
            g = graphs_all[c_idx]
            _, _, _, (attn1, attn2) = model_final(g, return_attention=True)
            attn1_ei, attn1_w = attn1
            n = g.num_nodes
            party = features[c][:, 2]
            cross_mask = np.abs(np.subtract.outer(party, party)) > 0.5
            same_mask = ~cross_mask
            np.fill_diagonal(same_mask, False)
            np.fill_diagonal(cross_mask, False)
            attn_mat = np.zeros((n, n))
            for e in range(attn1_ei.shape[1]):
                s, d = attn1_ei[0, e].item(), attn1_ei[1, e].item()
                if s < n and d < n:
                    attn_mat[s, d] = attn1_w[e].mean().item()
            cross_attn = float(attn_mat[cross_mask].mean()) if cross_mask.sum() > 0 else 0
            same_attn = float(attn_mat[same_mask].mean()) if same_mask.sum() > 0 else 0
            attention_analysis[c] = {
                'cross_party_attention': cross_attn,
                'same_party_attention': same_attn,
                'attention_ratio': float(same_attn / max(cross_attn, 1e-8)),
            }
            print(f"  {c}: same={same_attn:.6f} cross={cross_attn:.6f}", flush=True)

    all_results['attention_analysis'] = attention_analysis

    per_congress = {}
    for c_idx, c in enumerate(TEST_CONGRESSES):
        m = evaluate_model(model_final, [graphs_all[11 + c_idx]])
        per_congress[c] = m
        print(f"  Test {c}: Acc={m['accuracy']:.4f} F1={m['f1']:.4f} AUC={m['auc']:.4f}", flush=True)
    all_results['per_congress_test'] = per_congress

    with open(os.path.join(RESULTS_DIR, 'experiment_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nDone! Results in {RESULTS_DIR}", flush=True)
    return all_results

if __name__ == '__main__':
    run_full_experiment()
