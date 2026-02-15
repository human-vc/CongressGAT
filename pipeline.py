#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONGRESSES = list(range(100, 119))


def load_congress(congress_num):
    votes_path = os.path.join(DATA_DIR, f"H{congress_num}_votes.csv")
    members_path = os.path.join(DATA_DIR, f"H{congress_num}_members.csv")
    if not os.path.exists(votes_path) or not os.path.exists(members_path):
        return None, None
    votes = pd.read_csv(votes_path, low_memory=False)
    members = pd.read_csv(members_path, low_memory=False)
    return votes, members


def build_agreement_graph(votes, members, min_votes=50, min_shared=20):
    house_votes = votes[votes['chamber'] == 'House'].copy()
    house_votes = house_votes[house_votes['cast_code'].isin([1, 2, 3, 4, 5, 6])].copy()
    house_votes['yea'] = house_votes['cast_code'].isin([1, 2, 3]).astype(int)

    house_members = members[(members['chamber'] == 'House')].copy()
    house_members = house_members[house_members['party_code'].isin([100, 200])].copy()

    valid_icpsrs = set(house_members['icpsr'].unique())
    house_votes = house_votes[house_votes['icpsr'].isin(valid_icpsrs)]

    vote_counts = house_votes.groupby('icpsr').size()
    active = vote_counts[vote_counts >= min_votes].index.tolist()
    house_votes = house_votes[house_votes['icpsr'].isin(active)]

    member_list = sorted(house_votes['icpsr'].unique())
    member_to_idx = {m: i for i, m in enumerate(member_list)}
    n = len(member_list)

    rollcalls = sorted(house_votes['rollnumber'].unique())
    roll_to_idx = {r: i for i, r in enumerate(rollcalls)}

    vote_matrix = np.full((n, len(rollcalls)), np.nan)
    for _, row in house_votes.iterrows():
        i = member_to_idx.get(row['icpsr'])
        j = roll_to_idx.get(row['rollnumber'])
        if i is not None and j is not None:
            vote_matrix[i, j] = row['yea']

    agreement = np.zeros((n, n))
    shared_counts = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            mask = ~np.isnan(vote_matrix[i]) & ~np.isnan(vote_matrix[j])
            shared = mask.sum()
            if shared >= min_shared:
                agree = (vote_matrix[i, mask] == vote_matrix[j, mask]).sum()
                agreement[i, j] = agree / shared
                agreement[j, i] = agreement[i, j]
                shared_counts[i, j] = shared
                shared_counts[j, i] = shared

    member_info = {}
    for icpsr in member_list:
        row = house_members[house_members['icpsr'] == icpsr].iloc[0]
        member_info[int(icpsr)] = {
            'party': int(row['party_code']),
            'state': str(row['state_abbrev']),
            'name': str(row['bioname']),
            'nominate_dim1': float(row['nominate_dim1']) if pd.notna(row['nominate_dim1']) else 0.0,
            'nominate_dim2': float(row['nominate_dim2']) if pd.notna(row['nominate_dim2']) else 0.0,
        }

    return agreement, shared_counts, member_list, member_info, vote_matrix


def compute_spectral_features(agreement, threshold=0.5):
    n = agreement.shape[0]
    adj = (agreement > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    degrees = adj.sum(axis=1)
    isolated = degrees == 0
    if isolated.sum() > 0:
        adj_clean = adj[~isolated][:, ~isolated]
        degrees_clean = adj_clean.sum(axis=1)
    else:
        adj_clean = adj
        degrees_clean = degrees

    n_clean = adj_clean.shape[0]
    if n_clean < 3:
        return {'fiedler': 0.0, 'spectral_gap': 0.0, 'algebraic_connectivity': 0.0}

    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees_clean, 1e-10)))
    L_norm = np.eye(n_clean) - D_inv_sqrt @ adj_clean @ D_inv_sqrt

    try:
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
        fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        spectral_gap = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    except:
        fiedler = 0.0
        spectral_gap = 0.0

    return {
        'fiedler': fiedler,
        'spectral_gap': spectral_gap,
        'n_nodes': int(n),
        'n_edges': int(adj.sum() / 2),
        'density': float(adj.sum() / (n * (n - 1))) if n > 1 else 0.0,
        'mean_degree': float(degrees.mean()),
    }


def compute_polarization_metrics(member_info, member_list):
    dem_scores = []
    rep_scores = []
    for icpsr in member_list:
        info = member_info.get(int(icpsr), {})
        nom = info.get('nominate_dim1', 0.0)
        if info.get('party') == 100:
            dem_scores.append(nom)
        elif info.get('party') == 200:
            rep_scores.append(nom)

    if not dem_scores or not rep_scores:
        return {'party_distance': 0.0, 'overlap': 0.0}

    dem_mean = np.mean(dem_scores)
    rep_mean = np.mean(rep_scores)
    dem_std = np.std(dem_scores) if len(dem_scores) > 1 else 0.1
    rep_std = np.std(rep_scores) if len(rep_scores) > 1 else 0.1

    party_distance = abs(rep_mean - dem_mean)

    dem_max = max(dem_scores)
    rep_min = min(rep_scores)
    overlap = max(0, dem_max - rep_min)

    return {
        'party_distance': float(party_distance),
        'dem_mean': float(dem_mean),
        'rep_mean': float(rep_mean),
        'dem_std': float(dem_std),
        'rep_std': float(rep_std),
        'overlap': float(overlap),
        'n_dem': len(dem_scores),
        'n_rep': len(rep_scores),
    }


def compute_defection_labels(vote_matrix, member_list, member_info, threshold_pct=0.10):
    n_members = len(member_list)
    party_labels = []
    for icpsr in member_list:
        info = member_info.get(int(icpsr), {})
        party_labels.append(info.get('party', 0))

    party_labels = np.array(party_labels)
    defection_rates = np.zeros(n_members)

    for i in range(n_members):
        votes_i = vote_matrix[i]
        valid = ~np.isnan(votes_i)
        if valid.sum() == 0:
            continue

        same_party = party_labels == party_labels[i]
        same_party[i] = False

        if same_party.sum() == 0:
            continue

        party_votes = vote_matrix[same_party]
        defection_count = 0
        total_count = 0

        for j in range(vote_matrix.shape[1]):
            if np.isnan(votes_i[j]):
                continue
            party_valid = ~np.isnan(party_votes[:, j])
            if party_valid.sum() < 5:
                continue
            party_majority = np.nanmean(party_votes[:, j][party_valid]) > 0.5
            my_vote = votes_i[j] > 0.5
            if my_vote != party_majority:
                defection_count += 1
            total_count += 1

        if total_count > 0:
            defection_rates[i] = defection_count / total_count

    labels = (defection_rates >= threshold_pct).astype(int)
    return defection_rates, labels


def build_node_features(member_list, member_info, agreement, vote_matrix):
    n = len(member_list)
    features = []
    for i, icpsr in enumerate(member_list):
        info = member_info.get(int(icpsr), {})
        nom1 = info.get('nominate_dim1', 0.0)
        nom2 = info.get('nominate_dim2', 0.0)
        party = 1.0 if info.get('party') == 200 else 0.0

        valid_votes = ~np.isnan(vote_matrix[i])
        participation = valid_votes.sum() / vote_matrix.shape[1]
        yea_rate = np.nanmean(vote_matrix[i][valid_votes]) if valid_votes.sum() > 0 else 0.5

        mean_agreement = agreement[i].mean()
        cross_party_agreement = 0.0
        same_party_agreement = 0.0
        cross_count = 0
        same_count = 0
        for j, other_icpsr in enumerate(member_list):
            if i == j:
                continue
            other_info = member_info.get(int(other_icpsr), {})
            if other_info.get('party') != info.get('party'):
                cross_party_agreement += agreement[i, j]
                cross_count += 1
            else:
                same_party_agreement += agreement[i, j]
                same_count += 1

        cross_party_agreement = cross_party_agreement / max(cross_count, 1)
        same_party_agreement = same_party_agreement / max(same_count, 1)

        features.append([
            nom1, nom2, party, participation, yea_rate,
            mean_agreement, cross_party_agreement, same_party_agreement
        ])

    return np.array(features)


def process_all_congresses():
    all_results = {}

    for congress in CONGRESSES:
        print(f"\n{'='*60}")
        print(f"Processing Congress {congress}")
        print(f"{'='*60}")

        votes, members = load_congress(congress)
        if votes is None:
            print(f"  Skipping: data not found")
            continue

        try:
            agreement, shared, member_list, member_info, vote_matrix = build_agreement_graph(votes, members)
        except Exception as e:
            print(f"  Error building graph: {e}")
            continue

        print(f"  Members: {len(member_list)}, Rollcalls: {vote_matrix.shape[1]}")

        spectral = compute_spectral_features(agreement)
        print(f"  Fiedler: {spectral['fiedler']:.4f}, Edges: {spectral['n_edges']}")

        polarization = compute_polarization_metrics(member_info, member_list)
        print(f"  Party distance: {polarization['party_distance']:.4f}")

        node_features = build_node_features(member_list, member_info, agreement, vote_matrix)

        defection_results = {}
        for thresh in [0.05, 0.10, 0.15, 0.20, 0.25]:
            rates, labels = compute_defection_labels(vote_matrix, member_list, member_info, thresh)
            defection_results[f"thresh_{int(thresh*100)}"] = {
                'n_defectors': int(labels.sum()),
                'pct_defectors': float(labels.mean()),
            }
            print(f"  Defection@{int(thresh*100)}%: {labels.sum()} members ({labels.mean()*100:.1f}%)")

        congress_data = {
            'congress': congress,
            'n_members': len(member_list),
            'n_rollcalls': int(vote_matrix.shape[1]),
            'spectral': spectral,
            'polarization': polarization,
            'defection': defection_results,
        }
        all_results[congress] = congress_data

        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"congress_{congress}.npz"),
            agreement=agreement,
            vote_matrix=vote_matrix,
            node_features=node_features,
            member_list=np.array(member_list),
        )

        defection_rates_10, defection_labels_10 = compute_defection_labels(
            vote_matrix, member_list, member_info, 0.10
        )
        np.save(
            os.path.join(RESULTS_DIR, f"defection_rates_{congress}.npy"),
            defection_rates_10
        )
        np.save(
            os.path.join(RESULTS_DIR, f"defection_labels_{congress}.npy"),
            defection_labels_10
        )

        with open(os.path.join(RESULTS_DIR, f"member_info_{congress}.json"), 'w') as f:
            json.dump({str(k): v for k, v in member_info.items()}, f)

    with open(os.path.join(RESULTS_DIR, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nProcessed {len(all_results)} congresses")
    return all_results


if __name__ == "__main__":
    results = process_all_congresses()
