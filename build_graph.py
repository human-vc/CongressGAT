#!/usr/bin/env python3
"""
CongressGAT: Improved graph construction with lower threshold
"""

import pandas as pd
import numpy as np
from scipy import sparse
import json
import warnings
warnings.filterwarnings('ignore')

print("Loading data...")
votes_df = pd.read_csv('H116_votes.csv')
members_df = pd.read_csv('H116_members.csv')

# Filter to House and actual votes
votes_df = votes_df[votes_df['chamber'] == 'House'].copy()
votes_df = votes_df[votes_df['cast_code'].isin([1, 6])]

print(f"Votes: {len(votes_df)}, Members: {len(members_df)}")

# Create member lookup
member_info = {}
for _, row in members_df.iterrows():
    member_info[row['icpsr']] = {
        'name': row['bioname'],
        'party': row['party_code'],
        'state': row['state_abbrev'],
        'nominate_dim1': row['nominate_dim1']
    }

# Get active members (voted on at least 50 bills)
member_vote_counts = votes_df.groupby('icpsr').size()
active_members = member_vote_counts[member_vote_counts >= 50].index.tolist()
active_members = [m for m in active_members if m in member_info]
print(f"Active members (50+ votes): {len(active_members)}")

# Create mappings
idx_to_member = {i: m for i, m in enumerate(active_members)}
member_to_idx = {m: i for i, m in enumerate(active_members)}
n_members = len(active_members)

# Get roll calls
rollcalls = votes_df['rollnumber'].unique()
rollcall_to_idx = {r: i for i, r in enumerate(rollcalls)}

# Build vote matrix
print("Building vote matrix...")
vote_matrix = np.zeros((n_members, len(rollcalls)))

for _, row in votes_df.iterrows():
    icpsr = row['icpsr']
    if icpsr not in member_to_idx:
        continue
    rollnum = row['rollnumber']
    cast = row['cast_code']
    idx = member_to_idx[icpsr]
    col = rollcall_to_idx[rollnum]
    vote_matrix[idx, col] = 1 if cast == 1 else -1

print(f"Vote matrix: {n_members} x {len(rollcalls)}")

# Compute agreement using cosine similarity style
# Agreement when both vote same direction (both Yea or both Nay)
yea = (vote_matrix > 0).astype(float)
nay = (vote_matrix < 0).astype(float)

# Agreement = (both yea) + (both nay), normalized by (at least one voted)
both_voted = ((vote_matrix != 0).astype(int) @ (vote_matrix != 0).T.astype(int))
both_yea = yea @ yea.T
both_nay = nay @ nay.T
agree_counts = both_yea + both_nay

agreement_rate = np.where(both_voted > 0, agree_counts / both_voted, 0)
np.fill_diagonal(agreement_rate, 0)

print(f"Agreement range: {agreement_rate.min():.3f} - {agreement_rate.max():.3f}")
print(f"Mean agreement (non-zero): {agreement_rate[agreement_rate > 0].mean():.3f}")

# Use lower threshold based on distribution
TAU = 0.3  # Lower threshold
adjacency = (agreement_rate >= TAU).astype(float)
np.fill_diagonal(adjacency, 0)

# Remove isolated nodes
connected = np.sum(adjacency, axis=1) > 0
print(f"Connected nodes: {np.sum(connected)}/{n_members}")

# Get largest connected component
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

adj_sparse = csr_matrix(adjacency)
n_components, labels = connected_components(adj_sparse, directed=False)
component_sizes = [(labels == i).sum() for i in range(n_components)]
largest_component = np.argmax(component_sizes)

# Keep only largest component
keep = labels == largest_component
print(f"Largest component: {component_sizes[largest_component]} nodes")

# Rebuild for largest component
n_members = component_sizes[largest_component]
active_members = [active_members[i] for i in range(len(active_members)) if keep[i]]
idx_to_member = {i: m for i, m in enumerate(active_members)}
member_to_idx = {m: i for i, m in enumerate(active_members)}

# Rebuild matrices
vote_matrix = vote_matrix[keep][:, keep.shape[1]:] if keep.ndim > 1 else vote_matrix[keep]
# Actually rebuild from scratch
vote_matrix = np.zeros((n_members, len(rollcalls)))
for _, row in votes_df.iterrows():
    icpsr = row['icpsr']
    if icpsr not in member_to_idx:
        continue
    idx = member_to_idx[icpsr]
    col = rollcall_to_idx[row['rollnumber']]
    vote_matrix[idx, col] = 1 if row['cast_code'] == 1 else -1

yea = (vote_matrix > 0).astype(float)
nay = (vote_matrix < 0).astype(float)
both_voted = ((vote_matrix != 0).astype(int) @ (vote_matrix != 0).T.astype(int))
agree_counts = yea @ yea.T + nay @ nay.T
agreement_rate = np.where(both_voted > 0, agree_counts / both_voted, 0)
np.fill_diagonal(agreement_rate, 0)

adjacency = (agreement_rate >= TAU).astype(float)
np.fill_diagonal(adjacency, 0)

print(f"Final graph: {n_members} nodes")

# Spectral analysis
degree = np.sum(adjacency, axis=1)
D = np.diag(degree)
L = D - adjacency
D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degree, 1e-10)))
L_normalized = D_inv_sqrt @ L @ D_inv_sqrt

eigenvalues, eigenvectors = np.linalg.eigh(L_normalized)
fiedler_value = eigenvalues[1]
lambda_max = eigenvalues[-1]
polarization_index = 1 - (fiedler_value / lambda_max)

print(f"\n=== SPECTRAL RESULTS ===")
print(f"Fiedler value: {fiedler_value:.6f}")
print(f"Polarization: {polarization_index:.6f}")

# Fiedler bipartition
fiedler_vector = eigenvectors[:, 1]
pos_idx = np.where(fiedler_vector > 0)[0]
neg_idx = np.where(fiedler_vector < 0)[0]

def get_party(i):
    return member_info.get(idx_to_member.get(i, 0), {}).get('party', 0)

pos_parties = [get_party(i) for i in pos_idx]
neg_parties = [get_party(i) for i in neg_idx]

from collections import Counter
print(f"Positive: {Counter(pos_parties)}")
print(f"Negative: {Counter(neg_parties)}")

dem_pos = pos_parties.count(100)
rep_pos = pos_parties.count(200)
dem_neg = neg_parties.count(100)
rep_neg = neg_parties.count(200)
accuracy = (dem_pos + rep_neg) / (len(pos_parties) + len(neg_parties))
print(f"Party alignment: {accuracy*100:.1f}%")

# Network stats
n_edges = int(np.sum(adjacency) / 2)
cross = within = 0
for i in range(n_members):
    for j in range(i+1, n_members):
        if adjacency[i, j]:
            p1, p2 = get_party(i), get_party(j)
            if p1 in [100, 200] and p2 in [100, 200]:
                if p1 == p2:
                    within += 1
                else:
                    cross += 1

print(f"\nEdges: {n_edges}, Within: {within}, Cross: {cross}, Ratio: {within/max(cross,1):.2f}")

# Save
results = {
    'congress': 116,
    'fiedler_value': float(fiedler_value),
    'polarization_index': float(polarization_index),
    'n_members': n_members,
    'n_edges': n_edges,
    'party_alignment': accuracy
}
with open('spectral_results.json', 'w') as f:
    json.dump(results, f, indent=2)

sparse.save_npz('adjacency.npz', csr_matrix(adjacency))
sparse.save_npz('agreement.npz', csr_matrix(agreement_rate))

# Node features: [DW-NOMINATE, is_Dem, is_Rep]
features = np.array([[member_info[idx_to_member[i]]['nominate_dim1'] or 0,
                      1 if get_party(i) == 100 else 0,
                      1 if get_party(i) == 200 else 0] for i in range(n_members)])
np.save('node_features.npy', features)

with open('member_mapping.json', 'w') as f:
    json.dump({'idx_to_member': {str(k): int(v) for k, v in idx_to_member.items()}}, f)

print("\nDone!")
