#!/usr/bin/env python3
"""
CongressGAT Pipeline: Graph Attention Networks for Congressional Polarization Analysis
Builds per-congress graphs from Voteview roll-call data, trains a temporal GAT model,
and evaluates on three prediction tasks with proper train/test splits.
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    r2_score, f1_score, roc_auc_score, accuracy_score,
    precision_score, recall_score, confusion_matrix
)
from scipy.sparse import csr_matrix
from collections import defaultdict
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings('ignore')
np.random.seed(42)
torch.manual_seed(42)

DATA_DIR = os.path.expanduser('~/CongressionalGNN')
OUTPUT_DIR = os.path.expanduser('~/CongressionalGNN/results')
FIG_DIR = os.path.expanduser('~/CongressionalGNN/figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

CONGRESSES = list(range(110, 117))
TRAIN_CONGRESSES = [110, 111, 112, 113, 114]
TEST_CONGRESSES = [115, 116]

# ============================================================
# PART 1: Data Loading and Graph Construction
# ============================================================

def load_congress_data(congress_num):
    """Load members and votes for a single Congress from Voteview CSVs."""
    members_file = os.path.join(DATA_DIR, f'H{congress_num}_members.csv')
    votes_file = os.path.join(DATA_DIR, f'H{congress_num}_votes.csv')
    
    members = pd.read_csv(members_file)
    votes = pd.read_csv(votes_file)
    
    # Filter to House only, exclude President
    members = members[members['chamber'] == 'House'].copy()
    votes = votes[votes['chamber'] == 'House'].copy()
    
    return members, votes


def compute_agreement_matrix(members, votes):
    """
    Compute pairwise voting agreement between House members.
    Agreement = fraction of shared votes where both members voted the same way.
    cast_code: 1-3 = Yea, 4-6 = Nay, 7-9 = not voting/absent
    """
    icpsr_list = members['icpsr'].unique()
    icpsr_to_idx = {ic: i for i, ic in enumerate(icpsr_list)}
    n = len(icpsr_list)
    
    # Filter to substantive votes only (1-6)
    votes_sub = votes[votes['cast_code'].between(1, 6)].copy()
    # Binarize: 1-3 = Yea (1), 4-6 = Nay (0)
    votes_sub['vote_binary'] = (votes_sub['cast_code'] <= 3).astype(int)
    
    # Pivot to member x rollcall matrix
    # Only keep members in our list
    votes_sub = votes_sub[votes_sub['icpsr'].isin(icpsr_to_idx)]
    
    pivot = votes_sub.pivot_table(
        index='icpsr', columns='rollnumber', values='vote_binary', aggfunc='first'
    )
    
    # Reindex to ensure all members are present
    pivot = pivot.reindex(icpsr_list)
    
    vote_matrix = pivot.values.astype(np.float32)  # (n_members, n_rollcalls)
    valid = (~np.isnan(vote_matrix)).astype(np.float32)
    V = np.nan_to_num(vote_matrix, nan=0.0).astype(np.float32)
    
    # Masked votes: V where valid, 0 where missing
    V_masked = V * valid
    # Agreement = (both yea) + (both nay), only counting shared votes
    agree_yea = V_masked @ V_masked.T
    agree_nay = (valid - V_masked) @ (valid - V_masked).T
    agree = agree_yea + agree_nay
    shared = valid @ valid.T
    
    shared[shared == 0] = 1
    agreement = agree / shared
    np.fill_diagonal(agreement, 0)
    
    return agreement, icpsr_list, icpsr_to_idx


def build_graph_from_agreement(agreement, threshold=0.5):
    """
    Build adjacency matrix from agreement matrix.
    Edge exists if agreement > threshold.
    Edge weight = agreement score.
    """
    adj = (agreement > threshold).astype(float)
    adj_weighted = agreement * adj
    return adj, adj_weighted


def extract_node_features(members, icpsr_list):
    """
    Extract node features for each member:
    - DW-NOMINATE dim1 (ideology)
    - DW-NOMINATE dim2 (social/cultural)
    - Party (0=Democrat, 1=Republican, 2=other)
    - Tenure proxy (normalized ICPSR number within congress, lower = more senior)
    - State encoding (one-hot or numerical)
    """
    members_indexed = members.set_index('icpsr')
    
    features = []
    party_labels = []
    member_info = []
    
    for ic in icpsr_list:
        if ic in members_indexed.index:
            row = members_indexed.loc[ic]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            
            nom1 = row.get('nominate_dim1', 0) if pd.notna(row.get('nominate_dim1', np.nan)) else 0
            nom2 = row.get('nominate_dim2', 0) if pd.notna(row.get('nominate_dim2', np.nan)) else 0
            
            party_code = row.get('party_code', 0)
            if party_code == 100:
                party = 0  # Democrat
            elif party_code == 200:
                party = 1  # Republican
            else:
                party = 2  # Other/Independent
            
            state = row.get('state_abbrev', 'XX')
            bioname = row.get('bioname', 'Unknown')
            
            features.append([nom1, nom2, party])
            party_labels.append(party)
            member_info.append({
                'icpsr': int(ic),
                'name': bioname,
                'party': party_code,
                'state': state,
                'nominate_dim1': nom1,
                'nominate_dim2': nom2
            })
        else:
            features.append([0, 0, 0])
            party_labels.append(0)
            member_info.append({'icpsr': int(ic), 'name': 'Unknown', 'party': 0, 'state': 'XX'})
    
    return np.array(features, dtype=np.float32), np.array(party_labels), member_info


def compute_party_alignment(agreement, party_labels):
    """
    Compute spectral partitioning accuracy (party alignment from graph structure).
    Uses Fiedler vector of the graph Laplacian.
    """
    n = agreement.shape[0]
    degree = agreement.sum(axis=1)
    D = np.diag(degree)
    L = D - agreement
    
    # Compute Fiedler vector (second smallest eigenvalue)
    eigenvalues, eigenvectors = np.linalg.eigh(L)
    fiedler_vec = eigenvectors[:, 1]
    
    # Assign clusters based on sign of Fiedler vector
    clusters = (fiedler_vec > 0).astype(int)
    
    # Check both label assignments
    valid = party_labels < 2  # exclude independents
    if valid.sum() == 0:
        return 0.0, fiedler_vec, eigenvalues[1]
    
    acc1 = accuracy_score(party_labels[valid], clusters[valid])
    acc2 = accuracy_score(party_labels[valid], 1 - clusters[valid])
    
    return max(acc1, acc2), fiedler_vec, eigenvalues[1]


def compute_polarization_index(agreement, party_labels):
    """
    Compute polarization as ratio of within-party to between-party agreement.
    Higher values = more polarized.
    """
    n = len(party_labels)
    valid = party_labels < 2
    
    within_sum, within_count = 0, 0
    between_sum, between_count = 0, 0
    
    for i in range(n):
        if not valid[i]:
            continue
        for j in range(i+1, n):
            if not valid[j]:
                continue
            if party_labels[i] == party_labels[j]:
                within_sum += agreement[i, j]
                within_count += 1
            else:
                between_sum += agreement[i, j]
                between_count += 1
    
    within_avg = within_sum / max(within_count, 1)
    between_avg = between_sum / max(between_count, 1)
    
    # Polarization = 1 - (between/within); higher = more polarized
    polarization = 1 - (between_avg / max(within_avg, 1e-6))
    
    return polarization, within_avg, between_avg


# ============================================================
# PART 2: GAT Model
# ============================================================

class GATLayer(nn.Module):
    """Single Graph Attention Layer with multi-head attention."""
    
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.3, alpha=0.2, concat=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.n_heads = n_heads
        self.concat = concat
        self.dropout = dropout
        
        # Per-head linear transforms
        self.W = nn.Parameter(torch.zeros(n_heads, in_features, out_features))
        self.a_src = nn.Parameter(torch.zeros(n_heads, out_features, 1))
        self.a_dst = nn.Parameter(torch.zeros(n_heads, out_features, 1))
        
        self.leaky_relu = nn.LeakyReLU(alpha)
        self.softmax = nn.Softmax(dim=-1)
        
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)
    
    def forward(self, h, adj, return_attention=False):
        """
        h: (N, in_features) node features
        adj: (N, N) adjacency matrix (can be weighted)
        """
        N = h.size(0)
        
        # Transform features for each head: (n_heads, N, out_features)
        h_transformed = torch.einsum('nf,hfo->hno', h, self.W)
        
        # Compute attention scores
        attn_src = torch.einsum('hno,hoi->hni', h_transformed, self.a_src).squeeze(-1)  # (n_heads, N)
        attn_dst = torch.einsum('hno,hoi->hni', h_transformed, self.a_dst).squeeze(-1)  # (n_heads, N)
        
        # Pairwise attention: (n_heads, N, N)
        attn = attn_src.unsqueeze(2) + attn_dst.unsqueeze(1)
        attn = self.leaky_relu(attn)
        
        # Mask with adjacency
        mask = (adj > 0).float().unsqueeze(0)  # (1, N, N)
        attn = attn.masked_fill(mask == 0, float('-inf'))
        
        # Check for rows that are all -inf (isolated nodes)
        all_inf = (mask.sum(dim=-1) == 0).unsqueeze(-1).expand_as(attn)
        attn = attn.masked_fill(all_inf, 0)
        
        attn_weights = self.softmax(attn)  # (n_heads, N, N)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)
        
        # Weighted aggregation
        out = torch.einsum('hnm,hmo->hno', attn_weights, h_transformed)  # (n_heads, N, out_features)
        
        if self.concat:
            out = out.permute(1, 0, 2).contiguous().view(N, -1)  # (N, n_heads*out_features)
        else:
            out = out.mean(dim=0)  # (N, out_features)
        
        if return_attention:
            return out, attn_weights.mean(dim=0)  # average attention across heads
        return out


class TemporalSelfAttention(nn.Module):
    """Self-attention across time steps (Congresses)."""
    
    def __init__(self, embed_dim, n_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x):
        """
        x: (batch, T, embed_dim) - temporal sequence of graph-level embeddings
        Returns: (batch, T, embed_dim) attended embeddings
        """
        attn_out, attn_weights = self.attention(x, x, x)
        return self.norm(x + attn_out), attn_weights


class CongressGAT(nn.Module):
    """
    Full CongressGAT model:
    1. Per-congress GAT encoder
    2. Temporal self-attention across congresses
    3. Three prediction heads
    """
    
    def __init__(self, n_features=3, hidden_dim=32, gat_heads=4, temporal_heads=4, dropout=0.3):
        super().__init__()
        
        # GAT encoder (2 layers)
        self.gat1 = GATLayer(n_features, hidden_dim, n_heads=gat_heads, dropout=dropout, concat=True)
        gat1_out = hidden_dim * gat_heads
        self.gat2 = GATLayer(gat1_out, hidden_dim, n_heads=1, dropout=dropout, concat=False)
        
        self.bn1 = nn.BatchNorm1d(gat1_out)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        
        # Graph-level readout
        self.graph_proj = nn.Linear(hidden_dim, hidden_dim)
        
        # Temporal self-attention
        self.temporal_attn = TemporalSelfAttention(hidden_dim, n_heads=temporal_heads)
        
        # Prediction heads
        # (a) Polarization prediction (regression)
        self.polarization_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1)
        )
        
        # (b) Node-level party prediction (for coalition detection)
        self.coalition_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 2)
        )
        
        # (c) Defection prediction (binary: will member vote against party?)
        self.defection_head = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_graph(self, features, adj, return_attention=False):
        """Encode a single congress graph."""
        h = F.elu(self.gat1(features, adj, return_attention=False))
        h = self.bn1(h)
        h = self.dropout(h)
        
        if return_attention:
            h, attn_weights = self.gat2(h, adj, return_attention=True)
        else:
            h = self.gat2(h, adj)
            attn_weights = None
        
        h = self.bn2(h)
        node_embeddings = h  # (N, hidden_dim)
        
        # Graph-level embedding via mean pooling
        graph_embedding = self.graph_proj(h.mean(dim=0, keepdim=True))  # (1, hidden_dim)
        
        return node_embeddings, graph_embedding, attn_weights
    
    def forward(self, congress_data, return_attention=False):
        """
        congress_data: list of (features, adj) tuples, one per congress
        Returns predictions for each task
        """
        graph_embeddings = []
        all_node_embeddings = []
        all_attention = []
        
        for features, adj in congress_data:
            node_emb, graph_emb, attn = self.encode_graph(features, adj, return_attention=return_attention)
            all_node_embeddings.append(node_emb)
            graph_embeddings.append(graph_emb)
            if attn is not None:
                all_attention.append(attn)
        
        # Temporal attention over graph embeddings
        temporal_seq = torch.stack(graph_embeddings, dim=1)  # (1, T, hidden_dim) 
        temporal_out, temporal_attn = self.temporal_attn(temporal_seq)
        
        # Predictions
        # (a) Polarization per congress
        polarization_preds = []
        for t in range(temporal_out.size(1)):
            pred = self.polarization_head(temporal_out[0, t]).squeeze()
            polarization_preds.append(pred)
        polarization_preds = torch.stack(polarization_preds)
        
        # (b) Coalition detection per node per congress
        coalition_preds = []
        for node_emb in all_node_embeddings:
            pred = self.coalition_head(node_emb)  # (N, 2)
            coalition_preds.append(pred)
        
        # (c) Defection prediction per node per congress
        defection_preds = []
        for node_emb in all_node_embeddings:
            pred = self.defection_head(node_emb).squeeze(-1)  # (N,)
            defection_preds.append(pred)
        
        results = {
            'polarization': polarization_preds,
            'coalition': coalition_preds,
            'defection': defection_preds,
            'temporal_attention': temporal_attn,
            'node_embeddings': all_node_embeddings,
        }
        
        if return_attention:
            results['graph_attention'] = all_attention
        
        return results


# ============================================================
# PART 3: Defection Label Construction
# ============================================================

def compute_defection_labels(members, votes, icpsr_list):
    """
    Compute party defection rate for each member.
    Defection = voting against the majority of one's own party.
    Returns binary labels: 1 if defection rate > 10%, else 0.
    """
    members_indexed = members.set_index('icpsr')
    votes_sub = votes[votes['cast_code'].between(1, 6)].copy()
    votes_sub['vote_binary'] = (votes_sub['cast_code'] <= 3).astype(int)
    
    # Get party for each voter
    votes_sub = votes_sub[votes_sub['icpsr'].isin(icpsr_list)]
    votes_sub['party'] = votes_sub['icpsr'].map(
        members_indexed['party_code'].to_dict() if 'party_code' in members_indexed.columns 
        else {}
    )
    
    # For each roll call, compute party majority vote
    party_majority = votes_sub.groupby(['rollnumber', 'party'])['vote_binary'].mean()
    party_majority = (party_majority > 0.5).astype(int)
    party_majority = party_majority.reset_index()
    party_majority.columns = ['rollnumber', 'party', 'party_majority_vote']
    
    # Merge back
    votes_sub = votes_sub.merge(party_majority, on=['rollnumber', 'party'], how='left')
    
    # Defection: voted differently from party majority
    votes_sub['defected'] = (votes_sub['vote_binary'] != votes_sub['party_majority_vote']).astype(int)
    
    # Compute defection rate per member
    defection_rate = votes_sub.groupby('icpsr')['defected'].mean()
    
    # Map to our ordered list
    labels = np.zeros(len(icpsr_list))
    rates = np.zeros(len(icpsr_list))
    for i, ic in enumerate(icpsr_list):
        rate = defection_rate.get(ic, 0)
        rates[i] = rate
        labels[i] = 1 if rate > 0.10 else 0  # >10% defection = defector
    
    return labels, rates


# ============================================================
# PART 4: Process All Congresses
# ============================================================

def process_all_congresses():
    """Process all congresses, building graphs and labels."""
    all_data = {}
    
    for cnum in CONGRESSES:
        print(f"\nProcessing Congress {cnum}...")
        members, votes = load_congress_data(cnum)
        
        # Build agreement matrix
        agreement, icpsr_list, icpsr_to_idx = compute_agreement_matrix(members, votes)
        print(f"  Members: {len(icpsr_list)}, Agreement matrix: {agreement.shape}")
        
        # Build graph (threshold = 0.5 for edge existence)
        adj_binary, adj_weighted = build_graph_from_agreement(agreement, threshold=0.5)
        edge_count = int(adj_binary.sum()) // 2
        print(f"  Edges: {edge_count}")
        
        # Extract features
        features, party_labels, member_info = extract_node_features(members, icpsr_list)
        
        # Compute spectral alignment
        alignment, fiedler, fiedler_val = compute_party_alignment(agreement, party_labels)
        print(f"  Party alignment from spectral clustering: {alignment:.3f}")
        
        # Compute polarization
        polarization, within_avg, between_avg = compute_polarization_index(agreement, party_labels)
        print(f"  Polarization index: {polarization:.3f} (within={within_avg:.3f}, between={between_avg:.3f})")
        
        # Compute defection labels
        defection_labels, defection_rates = compute_defection_labels(members, votes, icpsr_list)
        n_defectors = int(defection_labels.sum())
        print(f"  Defectors (>10% cross-party votes): {n_defectors}/{len(icpsr_list)} ({100*n_defectors/len(icpsr_list):.1f}%)")
        
        all_data[cnum] = {
            'agreement': agreement,
            'adj_binary': adj_binary,
            'adj_weighted': adj_weighted,
            'features': features,
            'party_labels': party_labels,
            'member_info': member_info,
            'icpsr_list': icpsr_list,
            'polarization': polarization,
            'within_avg': within_avg,
            'between_avg': between_avg,
            'alignment': alignment,
            'fiedler_vector': fiedler,
            'fiedler_value': fiedler_val,
            'defection_labels': defection_labels,
            'defection_rates': defection_rates,
        }
    
    return all_data


# ============================================================
# PART 5: Training
# ============================================================

def train_congress_gat(all_data, n_epochs=200, lr=0.005):
    """Train CongressGAT with proper train/test split."""
    
    device = torch.device('cpu')  # Use CPU for reproducibility
    
    # Prepare data
    train_congress_data = []
    train_polarizations = []
    train_party_labels = []
    train_defection_labels = []
    
    test_congress_data = []
    test_polarizations = []
    test_party_labels = []
    test_defection_labels = []
    
    for cnum in CONGRESSES:
        d = all_data[cnum]
        features_t = torch.FloatTensor(d['features']).to(device)
        adj_t = torch.FloatTensor(d['adj_weighted']).to(device)
        
        if cnum in TRAIN_CONGRESSES:
            train_congress_data.append((features_t, adj_t))
            train_polarizations.append(d['polarization'])
            train_party_labels.append(torch.LongTensor(d['party_labels']).to(device))
            train_defection_labels.append(torch.FloatTensor(d['defection_labels']).to(device))
        else:
            test_congress_data.append((features_t, adj_t))
            test_polarizations.append(d['polarization'])
            test_party_labels.append(torch.LongTensor(d['party_labels']).to(device))
            test_defection_labels.append(torch.FloatTensor(d['defection_labels']).to(device))
    
    train_pol_target = torch.FloatTensor(train_polarizations).to(device)
    test_pol_target = torch.FloatTensor(test_polarizations).to(device)
    
    # Initialize model
    model = CongressGAT(n_features=3, hidden_dim=32, gat_heads=4, temporal_heads=2, dropout=0.3)
    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # Loss weights
    pol_weight = 1.0
    coal_weight = 1.0
    def_weight = 1.0
    
    best_loss = float('inf')
    best_state = None
    patience = 30
    no_improve = 0
    
    train_losses = []
    
    for epoch in range(n_epochs):
        model.train()
        optimizer.zero_grad()
        
        results = model(train_congress_data)
        
        # (a) Polarization loss (MSE)
        pol_loss = F.mse_loss(results['polarization'], train_pol_target)
        
        # (b) Coalition loss (cross-entropy per node, excluding independents)
        coal_loss = 0
        for t, party_lab in enumerate(train_party_labels):
            valid = party_lab < 2
            if valid.sum() > 0:
                coal_loss += F.cross_entropy(results['coalition'][t][valid], party_lab[valid])
        coal_loss /= len(train_party_labels)
        
        # (c) Defection loss (binary cross-entropy)
        def_loss = 0
        for t, def_lab in enumerate(train_defection_labels):
            valid = train_party_labels[t] < 2
            if valid.sum() > 0:
                def_loss += F.binary_cross_entropy(
                    results['defection'][t][valid], 
                    def_lab[valid]
                )
        def_loss /= len(train_defection_labels)
        
        total_loss = pol_weight * pol_loss + coal_weight * coal_loss + def_weight * def_loss
        
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(total_loss.item())
        
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        
        if no_improve >= patience:
            print(f"  Early stopping at epoch {epoch}")
            break
        
        if epoch % 50 == 0:
            print(f"  Epoch {epoch}: total_loss={total_loss.item():.4f} "
                  f"(pol={pol_loss.item():.4f}, coal={coal_loss.item():.4f}, def={def_loss.item():.4f})")
    
    # Load best model
    model.load_state_dict(best_state)
    
    return model, train_losses


def evaluate_model(model, all_data, device='cpu'):
    """Evaluate model on all congresses, reporting train vs test metrics."""
    model.eval()
    
    all_congress_data = []
    all_polarizations = []
    all_party_labels = []
    all_defection_labels = []
    all_defection_rates = []
    
    for cnum in CONGRESSES:
        d = all_data[cnum]
        features_t = torch.FloatTensor(d['features']).to(device)
        adj_t = torch.FloatTensor(d['adj_weighted']).to(device)
        all_congress_data.append((features_t, adj_t))
        all_polarizations.append(d['polarization'])
        all_party_labels.append(torch.LongTensor(d['party_labels']).to(device))
        all_defection_labels.append(torch.FloatTensor(d['defection_labels']).to(device))
        all_defection_rates.append(d['defection_rates'])
    
    with torch.no_grad():
        results = model(all_congress_data, return_attention=True)
    
    metrics = {'train': {}, 'test': {}, 'per_congress': {}}
    
    # Separate train/test indices
    train_idx = [i for i, c in enumerate(CONGRESSES) if c in TRAIN_CONGRESSES]
    test_idx = [i for i, c in enumerate(CONGRESSES) if c in TEST_CONGRESSES]
    
    # (a) Polarization R²
    pol_preds = results['polarization'].cpu().numpy()
    pol_true = np.array(all_polarizations)
    
    metrics['train']['polarization_r2'] = r2_score(pol_true[train_idx], pol_preds[train_idx])
    metrics['test']['polarization_r2'] = r2_score(pol_true[test_idx], pol_preds[test_idx])
    
    # (b) Coalition F1 and (c) Defection AUC
    for split_name, indices in [('train', train_idx), ('test', test_idx)]:
        all_coal_true, all_coal_pred = [], []
        all_def_true, all_def_prob = [], []
        
        for i in indices:
            # Coalition
            valid = all_party_labels[i] < 2
            if valid.sum() > 0:
                coal_pred = results['coalition'][i][valid].argmax(dim=1).cpu().numpy()
                coal_true = all_party_labels[i][valid].cpu().numpy()
                all_coal_true.extend(coal_true)
                all_coal_pred.extend(coal_pred)
            
            # Defection
            if valid.sum() > 0:
                def_prob = results['defection'][i][valid].cpu().numpy()
                def_true = all_defection_labels[i][valid].cpu().numpy()
                all_def_true.extend(def_true)
                all_def_prob.extend(def_prob)
        
        metrics[split_name]['coalition_f1'] = f1_score(all_coal_true, all_coal_pred, average='macro')
        metrics[split_name]['coalition_accuracy'] = accuracy_score(all_coal_true, all_coal_pred)
        
        if len(set(all_def_true)) > 1:
            metrics[split_name]['defection_auc'] = roc_auc_score(all_def_true, all_def_prob)
        else:
            metrics[split_name]['defection_auc'] = 0.5
    
    # Per-congress metrics
    for i, cnum in enumerate(CONGRESSES):
        valid = all_party_labels[i] < 2
        coal_pred = results['coalition'][i][valid].argmax(dim=1).cpu().numpy()
        coal_true = all_party_labels[i][valid].cpu().numpy()
        
        def_prob = results['defection'][i][valid].cpu().numpy()
        def_true = all_defection_labels[i][valid].cpu().numpy()
        
        per_c = {
            'polarization_true': float(pol_true[i]),
            'polarization_pred': float(pol_preds[i]),
            'coalition_f1': float(f1_score(coal_true, coal_pred, average='macro')),
            'coalition_accuracy': float(accuracy_score(coal_true, coal_pred)),
        }
        if len(set(def_true)) > 1:
            per_c['defection_auc'] = float(roc_auc_score(def_true, def_prob))
        else:
            per_c['defection_auc'] = 0.5
        
        metrics['per_congress'][cnum] = per_c
    
    # Store results for plotting
    metrics['predictions'] = {
        'polarization_pred': pol_preds.tolist(),
        'polarization_true': pol_true.tolist(),
    }
    
    # Store attention weights
    if 'graph_attention' in results and results['graph_attention']:
        metrics['graph_attention'] = [a.cpu().numpy() for a in results['graph_attention']]
    
    metrics['temporal_attention'] = results['temporal_attention'].cpu().numpy()
    metrics['node_embeddings'] = [e.cpu().numpy() for e in results['node_embeddings']]
    
    # Store coalition predictions and defection probabilities for plotting
    metrics['coalition_preds'] = []
    metrics['defection_probs'] = []
    for i in range(len(CONGRESSES)):
        metrics['coalition_preds'].append(results['coalition'][i].cpu().numpy())
        metrics['defection_probs'].append(results['defection'][i].cpu().numpy())
    
    return metrics


# ============================================================
# PART 6: Baselines
# ============================================================

def run_baselines(all_data):
    """Run baseline models: Logistic Regression, Random Forest, LSTM-like."""
    
    baseline_results = {}
    
    # Prepare features for baselines
    train_X, train_y_coal, train_y_def = [], [], []
    test_X, test_y_coal, test_y_def = [], [], []
    
    for cnum in CONGRESSES:
        d = all_data[cnum]
        valid = d['party_labels'] < 2
        features = d['features'][valid]
        parties = d['party_labels'][valid]
        defections = d['defection_labels'][valid]
        
        # Add agreement-based features
        agreement = d['agreement']
        # Mean agreement with same/different party
        mean_same = np.zeros(valid.sum())
        mean_diff = np.zeros(valid.sum())
        valid_idx = np.where(valid)[0]
        for ii, i in enumerate(valid_idx):
            same_mask = (d['party_labels'] == d['party_labels'][i]) & valid
            diff_mask = (d['party_labels'] != d['party_labels'][i]) & (d['party_labels'] < 2) & valid
            same_mask[i] = False
            if same_mask.sum() > 0:
                mean_same[ii] = agreement[i, same_mask].mean()
            if diff_mask.sum() > 0:
                mean_diff[ii] = agreement[i, diff_mask].mean()
        
        enriched = np.column_stack([features, mean_same, mean_diff])
        
        if cnum in TRAIN_CONGRESSES:
            train_X.append(enriched)
            train_y_coal.append(parties)
            train_y_def.append(defections)
        else:
            test_X.append(enriched)
            test_y_coal.append(parties)
            test_y_def.append(defections)
    
    train_X = np.vstack(train_X)
    train_y_coal = np.concatenate(train_y_coal)
    train_y_def = np.concatenate(train_y_def)
    test_X = np.vstack(test_X)
    test_y_coal = np.concatenate(test_y_coal)
    test_y_def = np.concatenate(test_y_def)
    
    scaler = StandardScaler()
    train_X_scaled = scaler.fit_transform(train_X)
    test_X_scaled = scaler.transform(test_X)
    
    # Logistic Regression
    lr_coal = LogisticRegression(max_iter=1000, random_state=42)
    lr_coal.fit(train_X_scaled, train_y_coal)
    lr_coal_train_pred = lr_coal.predict(train_X_scaled)
    lr_coal_test_pred = lr_coal.predict(test_X_scaled)
    
    lr_def = LogisticRegression(max_iter=1000, random_state=42)
    lr_def.fit(train_X_scaled, train_y_def)
    lr_def_train_prob = lr_def.predict_proba(train_X_scaled)[:, 1]
    lr_def_test_prob = lr_def.predict_proba(test_X_scaled)[:, 1]
    
    baseline_results['logistic_regression'] = {
        'train': {
            'coalition_f1': float(f1_score(train_y_coal, lr_coal_train_pred, average='macro')),
            'coalition_accuracy': float(accuracy_score(train_y_coal, lr_coal_train_pred)),
            'defection_auc': float(roc_auc_score(train_y_def, lr_def_train_prob)),
        },
        'test': {
            'coalition_f1': float(f1_score(test_y_coal, lr_coal_test_pred, average='macro')),
            'coalition_accuracy': float(accuracy_score(test_y_coal, lr_coal_test_pred)),
            'defection_auc': float(roc_auc_score(test_y_def, lr_def_test_prob)),
        }
    }
    
    # Random Forest
    rf_coal = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_coal.fit(train_X_scaled, train_y_coal)
    rf_coal_train_pred = rf_coal.predict(train_X_scaled)
    rf_coal_test_pred = rf_coal.predict(test_X_scaled)
    
    rf_def = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
    rf_def.fit(train_X_scaled, train_y_def)
    rf_def_train_prob = rf_def.predict_proba(train_X_scaled)[:, 1]
    rf_def_test_prob = rf_def.predict_proba(test_X_scaled)[:, 1]
    
    baseline_results['random_forest'] = {
        'train': {
            'coalition_f1': float(f1_score(train_y_coal, rf_coal_train_pred, average='macro')),
            'coalition_accuracy': float(accuracy_score(train_y_coal, rf_coal_train_pred)),
            'defection_auc': float(roc_auc_score(train_y_def, rf_def_train_prob)),
        },
        'test': {
            'coalition_f1': float(f1_score(test_y_coal, rf_coal_test_pred, average='macro')),
            'coalition_accuracy': float(accuracy_score(test_y_coal, rf_coal_test_pred)),
            'defection_auc': float(roc_auc_score(test_y_def, rf_def_test_prob)),
        }
    }
    
    # LSTM baseline (simple temporal model)
    # For polarization: use per-congress aggregate features to predict polarization
    pol_features = []
    pol_targets = []
    for cnum in CONGRESSES:
        d = all_data[cnum]
        valid = d['party_labels'] < 2
        feat = d['features'][valid]
        # Aggregate features: mean DW-NOM1, mean DW-NOM2, party ratio, mean agreement
        mean_nom1 = feat[:, 0].mean()
        mean_nom2 = feat[:, 1].mean()
        party_ratio = (d['party_labels'][valid] == 1).mean()
        mean_agree = d['agreement'][np.ix_(valid, valid)].mean()
        std_nom1 = feat[:, 0].std()
        
        pol_features.append([mean_nom1, mean_nom2, party_ratio, mean_agree, std_nom1])
        pol_targets.append(d['polarization'])
    
    pol_features = np.array(pol_features)
    pol_targets = np.array(pol_targets)
    
    # Simple linear regression for polarization baseline
    from sklearn.linear_model import LinearRegression
    lr_pol = LinearRegression()
    lr_pol.fit(pol_features[:5], pol_targets[:5])
    pol_train_pred = lr_pol.predict(pol_features[:5])
    pol_test_pred = lr_pol.predict(pol_features[5:])
    
    baseline_results['linear_regression_polarization'] = {
        'train_r2': float(r2_score(pol_targets[:5], pol_train_pred)),
        'test_r2': float(r2_score(pol_targets[5:], pol_test_pred)),
    }
    
    # Store feature importances from RF for interpretability
    baseline_results['rf_feature_importance'] = {
        'coalition': rf_coal.feature_importances_.tolist(),
        'defection': rf_def.feature_importances_.tolist(),
        'feature_names': ['DW-NOM1', 'DW-NOM2', 'Party', 'Mean Same-Party Agree', 'Mean Cross-Party Agree']
    }
    
    return baseline_results


# ============================================================
# PART 7: Interpretability Analysis
# ============================================================

def attention_analysis(model, all_data, metrics):
    """Analyze what GAT attention weights learn."""
    model.eval()
    device = torch.device('cpu')
    
    analysis = {}
    
    for idx, cnum in enumerate(CONGRESSES):
        d = all_data[cnum]
        features_t = torch.FloatTensor(d['features']).to(device)
        adj_t = torch.FloatTensor(d['adj_weighted']).to(device)
        
        with torch.no_grad():
            node_emb, graph_emb, attn_weights = model.encode_graph(features_t, adj_t, return_attention=True)
        
        if attn_weights is not None:
            attn = attn_weights.cpu().numpy()
            party = d['party_labels']
            valid = party < 2
            
            # Analyze attention patterns: same-party vs cross-party
            n = len(party)
            same_party_attn = []
            cross_party_attn = []
            
            for i in range(n):
                if not valid[i]:
                    continue
                for j in range(n):
                    if i == j or not valid[j]:
                        continue
                    if attn[i, j] > 0:
                        if party[i] == party[j]:
                            same_party_attn.append(attn[i, j])
                        else:
                            cross_party_attn.append(attn[i, j])
            
            analysis[cnum] = {
                'mean_same_party_attention': float(np.mean(same_party_attn)) if same_party_attn else 0,
                'mean_cross_party_attention': float(np.mean(cross_party_attn)) if cross_party_attn else 0,
                'same_party_count': len(same_party_attn),
                'cross_party_count': len(cross_party_attn),
                'attention_matrix': attn,
            }
            
            # Find top attention edges
            top_edges = []
            flat_idx = np.argsort(attn.flatten())[-20:]
            for fi in flat_idx:
                i, j = divmod(fi, n)
                if valid[i] and valid[j] and i != j:
                    top_edges.append({
                        'from': d['member_info'][i]['name'],
                        'to': d['member_info'][j]['name'],
                        'attention': float(attn[i, j]),
                        'same_party': bool(party[i] == party[j]),
                    })
            analysis[cnum]['top_edges'] = top_edges
    
    return analysis


# ============================================================
# PART 8: Causal Analysis (Difference-in-Differences style)
# ============================================================

def causal_analysis(all_data):
    """
    Quasi-causal analysis of polarization shifts using difference-in-differences logic.
    Key events:
    - Tea Party wave: 111th -> 112th Congress (2010 midterms)  
    - Trump era: 114th -> 115th Congress (2016 election)
    """
    results = {}
    
    # Track polarization trajectory
    pol_trajectory = {cnum: all_data[cnum]['polarization'] for cnum in CONGRESSES}
    
    # Tea Party effect: compare 111->112 shift
    pol_111 = all_data[111]['polarization']
    pol_112 = all_data[112]['polarization']
    tea_party_shift = pol_112 - pol_111
    
    # Pre-treatment trend: 110->111
    pol_110 = all_data[110]['polarization']
    pre_trend = pol_111 - pol_110
    
    # Tea Party "effect" = actual shift minus pre-trend
    tea_party_effect = tea_party_shift - pre_trend
    
    results['tea_party'] = {
        'pre_trend_110_111': float(pre_trend),
        'shift_111_112': float(tea_party_shift),
        'estimated_effect': float(tea_party_effect),
        'polarization_110': float(pol_110),
        'polarization_111': float(pol_111),
        'polarization_112': float(pol_112),
    }
    
    # Track defection rates around Tea Party
    def_rates = {}
    for cnum in CONGRESSES:
        d = all_data[cnum]
        valid = d['party_labels'] < 2
        rep_mask = (d['party_labels'] == 1) & valid
        dem_mask = (d['party_labels'] == 0) & valid
        
        def_rates[cnum] = {
            'overall': float(d['defection_rates'][valid].mean()),
            'republican': float(d['defection_rates'][rep_mask].mean()) if rep_mask.sum() > 0 else 0,
            'democrat': float(d['defection_rates'][dem_mask].mean()) if dem_mask.sum() > 0 else 0,
        }
    
    results['defection_trajectory'] = def_rates
    
    # Trump era
    pol_114 = all_data[114]['polarization']
    pol_115 = all_data[115]['polarization']
    pol_116 = all_data[116]['polarization']
    pre_trend_trump = pol_114 - all_data[113]['polarization']
    trump_shift = pol_115 - pol_114
    
    results['trump_era'] = {
        'pre_trend_113_114': float(pre_trend_trump),
        'shift_114_115': float(trump_shift),
        'estimated_effect': float(trump_shift - pre_trend_trump),
        'polarization_114': float(pol_114),
        'polarization_115': float(pol_115),
        'polarization_116': float(pol_116),
    }
    
    # Freshman effect analysis: how do new members differ?
    for cnum in [112, 115]:
        prev = cnum - 1
        d_cur = all_data[cnum]
        d_prev = all_data[prev]
        
        cur_icpsr = set(d_cur['icpsr_list'])
        prev_icpsr = set(d_prev['icpsr_list'])
        
        new_members = cur_icpsr - prev_icpsr
        returning = cur_icpsr & prev_icpsr
        
        # Compare ideology of new vs returning members
        cur_idx_new = [i for i, ic in enumerate(d_cur['icpsr_list']) if ic in new_members]
        cur_idx_ret = [i for i, ic in enumerate(d_cur['icpsr_list']) if ic in returning]
        
        if cur_idx_new and cur_idx_ret:
            new_nom1 = d_cur['features'][cur_idx_new, 0]
            ret_nom1 = d_cur['features'][cur_idx_ret, 0]
            new_party = d_cur['party_labels'][cur_idx_new]
            
            # Among new Republican members
            new_rep = new_nom1[new_party == 1]
            ret_rep_mask = d_cur['party_labels'][cur_idx_ret] == 1
            ret_rep = ret_nom1[ret_rep_mask]
            
            results[f'freshman_effect_{cnum}'] = {
                'n_new_members': len(new_members),
                'n_returning': len(returning),
                'new_rep_mean_ideology': float(new_rep.mean()) if len(new_rep) > 0 else 0,
                'returning_rep_mean_ideology': float(ret_rep.mean()) if len(ret_rep) > 0 else 0,
                'ideology_shift': float(new_rep.mean() - ret_rep.mean()) if len(new_rep) > 0 and len(ret_rep) > 0 else 0,
            }
    
    results['polarization_trajectory'] = pol_trajectory
    
    return results


# ============================================================
# PART 9: Visualization
# ============================================================

def create_all_figures(all_data, metrics, baseline_results, causal_results, attention_results):
    """Generate all figures for the paper."""
    
    sns.set_style('whitegrid')
    plt.rcParams.update({
        'font.size': 11,
        'axes.titlesize': 13,
        'axes.labelsize': 11,
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'savefig.bbox': 'tight',
    })
    
    # ---- Figure 1: Polarization Over Time ----
    fig, ax = plt.subplots(figsize=(8, 5))
    congresses = CONGRESSES
    years = [2007, 2009, 2011, 2013, 2015, 2017, 2019]
    pol_true = [all_data[c]['polarization'] for c in congresses]
    pol_pred = metrics['predictions']['polarization_pred']
    
    ax.plot(years, pol_true, 'ko-', linewidth=2, markersize=8, label='Observed', zorder=5)
    ax.plot(years, pol_pred, 's--', color='#2196F3', linewidth=2, markersize=7, label='CongressGAT Predicted', zorder=4)
    
    # Mark train/test split
    ax.axvline(x=2016, color='gray', linestyle=':', alpha=0.7)
    ax.text(2016.1, max(pol_true)*0.95, 'Train | Test', fontsize=9, color='gray')
    
    # Mark key events
    ax.annotate('Tea Party\nWave', xy=(2011, pol_true[2]), xytext=(2012, pol_true[2]+0.08),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    
    ax.set_xlabel('Year (Start of Congress)')
    ax.set_ylabel('Polarization Index')
    ax.set_title('Congressional Polarization: Observed vs. Predicted')
    ax.legend(frameon=True)
    ax.set_xticks(years)
    ax.set_xticklabels([f'{y}\n({c}th)' for y, c in zip(years, congresses)], fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'polarization_over_time.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'polarization_over_time.png'))
    plt.close()
    
    # ---- Figure 2: Network Visualization (selected congresses) ----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, cnum in enumerate([110, 112, 116]):
        ax = axes[ax_idx]
        d = all_data[cnum]
        features = d['features']
        party = d['party_labels']
        
        # Use DW-NOMINATE dims as coordinates
        x = features[:, 0]
        y = features[:, 1]
        
        colors = ['#2196F3' if p == 0 else '#F44336' if p == 1 else '#9E9E9E' for p in party]
        
        # Draw a subset of edges (top agreement edges)
        agreement = d['agreement']
        # Only draw edges with agreement > 0.7
        edge_thresh = 0.7
        for i in range(len(party)):
            for j in range(i+1, len(party)):
                if agreement[i, j] > edge_thresh:
                    ax.plot([x[i], x[j]], [y[i], y[j]], 
                           color='gray', alpha=0.03, linewidth=0.3)
        
        ax.scatter(x, y, c=colors, s=15, alpha=0.7, edgecolors='white', linewidth=0.3, zorder=5)
        ax.set_title(f'{cnum}th Congress ({2007 + (cnum-110)*2})')
        ax.set_xlabel('DW-NOMINATE Dim. 1')
        if ax_idx == 0:
            ax.set_ylabel('DW-NOMINATE Dim. 2')
        ax.set_xlim(-0.8, 1.0)
        ax.set_ylim(-0.8, 1.0)
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#2196F3', markersize=8, label='Democrat'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#F44336', markersize=8, label='Republican'),
    ]
    axes[2].legend(handles=legend_elements, loc='lower right')
    
    plt.suptitle('Congressional Voting Networks in Ideological Space', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'network_visualization.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'network_visualization.png'))
    plt.close()
    
    # ---- Figure 3: Temporal Attention Heatmap ----
    fig, ax = plt.subplots(figsize=(7, 5))
    temporal_attn = metrics['temporal_attention']
    if temporal_attn.ndim == 3:
        temporal_attn = temporal_attn[0]  # batch dim
    
    congress_labels = [f'{c}th' for c in CONGRESSES]
    im = ax.imshow(temporal_attn, cmap='YlOrRd', aspect='auto')
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_yticks(range(len(CONGRESSES)))
    ax.set_xticklabels(congress_labels, fontsize=10)
    ax.set_yticklabels(congress_labels, fontsize=10)
    ax.set_xlabel('Key (Source Congress)')
    ax.set_ylabel('Query (Target Congress)')
    ax.set_title('Temporal Self-Attention Weights')
    plt.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'temporal_attention.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'temporal_attention.png'))
    plt.close()
    
    # ---- Figure 4: Attention Analysis - Same vs Cross Party ----
    fig, ax = plt.subplots(figsize=(8, 5))
    same_attn = [attention_results[c]['mean_same_party_attention'] for c in CONGRESSES if c in attention_results]
    cross_attn = [attention_results[c]['mean_cross_party_attention'] for c in CONGRESSES if c in attention_results]
    valid_congresses = [c for c in CONGRESSES if c in attention_results]
    
    x = np.arange(len(valid_congresses))
    width = 0.35
    ax.bar(x - width/2, same_attn, width, label='Same-Party Attention', color='#4CAF50', alpha=0.8)
    ax.bar(x + width/2, cross_attn, width, label='Cross-Party Attention', color='#FF9800', alpha=0.8)
    ax.set_xlabel('Congress')
    ax.set_ylabel('Mean Attention Weight')
    ax.set_title('GAT Attention: Within-Party vs. Cross-Party')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{c}th' for c in valid_congresses])
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'attention_party_analysis.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'attention_party_analysis.png'))
    plt.close()
    
    # ---- Figure 5: ROC-style Defection Analysis ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Defection rates over time
    ax = axes[0]
    def_traj = causal_results['defection_trajectory']
    rep_rates = [def_traj[c]['republican'] for c in CONGRESSES]
    dem_rates = [def_traj[c]['democrat'] for c in CONGRESSES]
    overall_rates = [def_traj[c]['overall'] for c in CONGRESSES]
    
    ax.plot(years, rep_rates, 'o-', color='#F44336', linewidth=2, label='Republican')
    ax.plot(years, dem_rates, 's-', color='#2196F3', linewidth=2, label='Democrat')
    ax.plot(years, overall_rates, '^--', color='gray', linewidth=1.5, label='Overall')
    ax.set_xlabel('Year')
    ax.set_ylabel('Mean Defection Rate')
    ax.set_title('Party Defection Rates Over Time')
    ax.legend()
    ax.set_xticks(years)
    ax.set_xticklabels([f'{y}' for y in years], fontsize=9)
    
    # Defection AUC comparison across methods
    ax = axes[1]
    gat_aucs = [metrics['per_congress'][c]['defection_auc'] for c in CONGRESSES]
    ax.bar(range(len(CONGRESSES)), gat_aucs, color=['#4CAF50' if c in TRAIN_CONGRESSES else '#FF9800' for c in CONGRESSES], alpha=0.8)
    ax.set_xticks(range(len(CONGRESSES)))
    ax.set_xticklabels([f'{c}th' for c in CONGRESSES])
    ax.set_ylabel('Defection AUC')
    ax.set_title('Defection Prediction AUC by Congress')
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'defection_analysis.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'defection_analysis.png'))
    plt.close()
    
    # ---- Figure 6: Model Comparison Bar Chart ----
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    
    models = ['CongressGAT', 'Logistic Reg.', 'Random Forest']
    
    # Coalition F1
    ax = axes[0]
    train_vals = [
        metrics['train']['coalition_f1'],
        baseline_results['logistic_regression']['train']['coalition_f1'],
        baseline_results['random_forest']['train']['coalition_f1'],
    ]
    test_vals = [
        metrics['test']['coalition_f1'],
        baseline_results['logistic_regression']['test']['coalition_f1'],
        baseline_results['random_forest']['test']['coalition_f1'],
    ]
    x = np.arange(len(models))
    ax.bar(x - 0.15, train_vals, 0.3, label='Train', color='#4CAF50', alpha=0.8)
    ax.bar(x + 0.15, test_vals, 0.3, label='Test', color='#FF9800', alpha=0.8)
    ax.set_ylabel('Macro F1')
    ax.set_title('Coalition Detection (F1)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Defection AUC
    ax = axes[1]
    train_vals = [
        metrics['train']['defection_auc'],
        baseline_results['logistic_regression']['train']['defection_auc'],
        baseline_results['random_forest']['train']['defection_auc'],
    ]
    test_vals = [
        metrics['test']['defection_auc'],
        baseline_results['logistic_regression']['test']['defection_auc'],
        baseline_results['random_forest']['test']['defection_auc'],
    ]
    ax.bar(x - 0.15, train_vals, 0.3, label='Train', color='#4CAF50', alpha=0.8)
    ax.bar(x + 0.15, test_vals, 0.3, label='Test', color='#FF9800', alpha=0.8)
    ax.set_ylabel('AUC')
    ax.set_title('Defection Prediction (AUC)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    
    # Polarization R²
    ax = axes[2]
    pol_models = ['CongressGAT', 'Linear Reg.']
    train_r2 = [
        metrics['train']['polarization_r2'],
        baseline_results['linear_regression_polarization']['train_r2'],
    ]
    test_r2 = [
        metrics['test']['polarization_r2'],
        baseline_results['linear_regression_polarization']['test_r2'],
    ]
    x2 = np.arange(len(pol_models))
    ax.bar(x2 - 0.15, train_r2, 0.3, label='Train', color='#4CAF50', alpha=0.8)
    ax.bar(x2 + 0.15, test_r2, 0.3, label='Test', color='#FF9800', alpha=0.8)
    ax.set_ylabel('R²')
    ax.set_title('Polarization Prediction (R²)')
    ax.set_xticks(x2)
    ax.set_xticklabels(pol_models, fontsize=9)
    ax.legend()
    
    plt.suptitle('Model Comparison: Train (110th-114th) vs. Test (115th-116th)', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'model_comparison.png'))
    plt.close()
    
    # ---- Figure 7: Causal Analysis - Tea Party and Trump Effects ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Tea Party
    ax = axes[0]
    tp = causal_results['tea_party']
    bars = [tp['polarization_110'], tp['polarization_111'], tp['polarization_112']]
    colors = ['#9E9E9E', '#9E9E9E', '#F44336']
    ax.bar([0, 1, 2], bars, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['110th\n(Pre)', '111th\n(Pre)', '112th\n(Post-Tea Party)'])
    ax.set_ylabel('Polarization Index')
    ax.set_title('Tea Party Effect on Polarization')
    ax.annotate(f'Δ = {tp["shift_111_112"]:.3f}', xy=(1.5, max(bars)*0.9), fontsize=11, 
               ha='center', color='red', fontweight='bold')
    
    # Trump era
    ax = axes[1]
    tr = causal_results['trump_era']
    bars = [tr['polarization_114'], tr['polarization_115'], tr['polarization_116']]
    colors = ['#9E9E9E', '#FF9800', '#FF9800']
    ax.bar([0, 1, 2], bars, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(['114th\n(Pre)', '115th\n(Trump Year 1-2)', '116th\n(Trump Year 3-4)'])
    ax.set_ylabel('Polarization Index')
    ax.set_title('Trump Era Effect on Polarization')
    ax.annotate(f'Δ = {tr["shift_114_115"]:.3f}', xy=(0.5, max(bars)*0.9), fontsize=11,
               ha='center', color='#FF9800', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'causal_analysis.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'causal_analysis.png'))
    plt.close()
    
    # ---- Figure 8: Embedding Visualization (t-SNE style using PCA for speed) ----
    from sklearn.decomposition import PCA
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax_idx, (cnum_idx, cnum) in enumerate([(0, 110), (2, 112), (6, 116)]):
        ax = axes[ax_idx]
        embeddings = metrics['node_embeddings'][cnum_idx]
        party = all_data[cnum]['party_labels']
        
        pca = PCA(n_components=2)
        emb_2d = pca.fit_transform(embeddings)
        
        colors = ['#2196F3' if p == 0 else '#F44336' if p == 1 else '#9E9E9E' for p in party]
        ax.scatter(emb_2d[:, 0], emb_2d[:, 1], c=colors, s=15, alpha=0.7, edgecolors='white', linewidth=0.3)
        ax.set_title(f'{cnum}th Congress Embeddings')
        ax.set_xlabel('PC 1')
        if ax_idx == 0:
            ax.set_ylabel('PC 2')
    
    plt.suptitle('GAT Node Embeddings (PCA Projection)', fontsize=14, y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'embeddings_pca.pdf'))
    plt.savefig(os.path.join(FIG_DIR, 'embeddings_pca.png'))
    plt.close()
    
    print(f"All figures saved to {FIG_DIR}/")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("CongressGAT Pipeline")
    print("=" * 60)
    
    # Step 1: Process all congresses
    print("\n[1/7] Processing congressional data...")
    all_data = process_all_congresses()
    
    # Step 2: Train model
    print("\n[2/7] Training CongressGAT model...")
    print(f"  Train: {TRAIN_CONGRESSES}, Test: {TEST_CONGRESSES}")
    model, train_losses = train_congress_gat(all_data, n_epochs=300, lr=0.005)
    
    # Step 3: Evaluate
    print("\n[3/7] Evaluating model...")
    metrics = evaluate_model(model, all_data)
    
    print(f"\n  TRAIN metrics:")
    print(f"    Polarization R²: {metrics['train']['polarization_r2']:.4f}")
    print(f"    Coalition F1:    {metrics['train']['coalition_f1']:.4f}")
    print(f"    Defection AUC:   {metrics['train']['defection_auc']:.4f}")
    print(f"\n  TEST metrics (out-of-sample):")
    print(f"    Polarization R²: {metrics['test']['polarization_r2']:.4f}")
    print(f"    Coalition F1:    {metrics['test']['coalition_f1']:.4f}")
    print(f"    Defection AUC:   {metrics['test']['defection_auc']:.4f}")
    
    # Step 4: Baselines
    print("\n[4/7] Running baselines...")
    baseline_results = run_baselines(all_data)
    
    print(f"\n  Logistic Regression (test):")
    print(f"    Coalition F1:  {baseline_results['logistic_regression']['test']['coalition_f1']:.4f}")
    print(f"    Defection AUC: {baseline_results['logistic_regression']['test']['defection_auc']:.4f}")
    print(f"  Random Forest (test):")
    print(f"    Coalition F1:  {baseline_results['random_forest']['test']['coalition_f1']:.4f}")
    print(f"    Defection AUC: {baseline_results['random_forest']['test']['defection_auc']:.4f}")
    print(f"  Linear Reg. Polarization (test R²): {baseline_results['linear_regression_polarization']['test_r2']:.4f}")
    
    # Step 5: Attention analysis
    print("\n[5/7] Analyzing attention weights...")
    attention_results = attention_analysis(model, all_data, metrics)
    for cnum in CONGRESSES:
        if cnum in attention_results:
            ar = attention_results[cnum]
            print(f"  Congress {cnum}: same-party attn={ar['mean_same_party_attention']:.4f}, "
                  f"cross-party attn={ar['mean_cross_party_attention']:.4f}")
    
    # Step 6: Causal analysis
    print("\n[6/7] Running causal analysis...")
    causal_results = causal_analysis(all_data)
    print(f"  Tea Party effect (111->112): {causal_results['tea_party']['estimated_effect']:.4f}")
    print(f"  Trump effect (114->115): {causal_results['trump_era']['estimated_effect']:.4f}")
    
    # Step 7: Create figures
    print("\n[7/7] Creating figures...")
    create_all_figures(all_data, metrics, baseline_results, causal_results, attention_results)
    
    # Save all results
    results_summary = {
        'train_metrics': metrics['train'],
        'test_metrics': metrics['test'],
        'per_congress': metrics['per_congress'],
        'predictions': metrics['predictions'],
        'baselines': baseline_results,
        'causal': causal_results,
        'attention_summary': {
            cnum: {k: v for k, v in attention_results[cnum].items() if k != 'attention_matrix'}
            for cnum in CONGRESSES if cnum in attention_results
        },
    }
    
    with open(os.path.join(OUTPUT_DIR, 'full_results.json'), 'w') as f:
        json.dump(results_summary, f, indent=2, default=str)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'congress_gat_model.pt'))
    
    print("\n" + "=" * 60)
    print("Pipeline complete!")
    print(f"Results saved to {OUTPUT_DIR}/")
    print(f"Figures saved to {FIG_DIR}/")
    print("=" * 60)
    
    return all_data, model, metrics, baseline_results, causal_results, attention_results


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(line_buffering=True)
    main()
