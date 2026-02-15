#!/usr/bin/env python3

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy import sparse
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
    n_rolls = len(rollcalls)

    vote_matrix = np.full((n, n_rolls), np.nan)
    for _, row in house_votes.iterrows():
        i = member_to_idx.get(row['icpsr'])
        j = roll_to_idx.get(row['rollnumber'])
        if i is not None and j is not None:
            vote_matrix[i, j] = row['yea']

    valid_mask = ~np.isnan(vote_matrix)
    vm_filled = np.nan_to_num(vote_matrix, nan=-1.0)

    both_valid = valid_mask.astype(np.float32) @ valid_mask.astype(np.float32).T

    vm_binary = np.where(valid_mask, vote_matrix, 0.0).astype(np.float32)
    both_yea = vm_binary @ vm_binary.T
    both_nay = (1.0 - vm_binary) * valid_mask.astype(np.float32)
    both_nay = both_nay @ both_nay.T

    agree_count = both_yea + both_nay
    agreement = np.divide(agree_count, both_valid, out=np.zeros_like(agree_count), where=both_valid >= min_shared)
    agreement[both_valid < min_shared] = 0.0
    np.fill_diagonal(agreement, 0.0)

    member_info = {}
    for icpsr in member_list:
        rows = house_members[house_members['icpsr'] == icpsr]
        if len(rows) == 0:
            continue
        row = rows.iloc[0]
        member_info[int(icpsr)] = {
            'party': int(row['party_code']),
            'state': str(row['state_abbrev']),
            'name': str(row['bioname']),
            'nominate_dim1': float(row['nominate_dim1']) if pd.notna(row['nominate_dim1']) else 0.0,
            'nominate_dim2': float(row['nominate_dim2']) if pd.notna(row['nominate_dim2']) else 0.0,
        }

    return agreement, both_valid, member_list, member_info, vote_matrix


def compute_spectral_features(agreement, threshold=0.5):
    n = agreement.shape[0]
    adj = (agreement > threshold).astype(float)
    np.fill_diagonal(adj, 0)

    degrees = adj.sum(axis=1)
    valid = degrees > 0
    adj_clean = adj[np.ix_(valid, valid)]
    degrees_clean = adj_clean.sum(axis=1)
    n_clean = adj_clean.shape[0]

    if n_clean < 3:
        return {'fiedler': 0.0, 'spectral_gap': 0.0, 'n_nodes': n, 'n_edges': 0, 'density': 0.0, 'mean_degree': 0.0}

    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees_clean, 1e-10)))
    L_norm = np.eye(n_clean) - D_inv_sqrt @ adj_clean @ D_inv_sqrt

    try:
        eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
        fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
    except:
        fiedler = 0.0

    return {
        'fiedler': fiedler,
        'spectral_gap': fiedler,
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

    return {
        'party_distance': float(abs(rep_mean - dem_mean)),
        'dem_mean': float(dem_mean),
        'rep_mean': float(rep_mean),
        'dem_std': float(np.std(dem_scores)) if len(dem_scores) > 1 else 0.1,
        'rep_std': float(np.std(rep_scores)) if len(rep_scores) > 1 else 0.1,
        'overlap': float(max(0, max(dem_scores) - min(rep_scores))),
        'n_dem': len(dem_scores),
        'n_rep': len(rep_scores),
    }


def compute_defection_labels(vote_matrix, member_list, member_info, threshold_pct=0.10):
    n_members = len(member_list)
    party_labels = np.array([
        member_info.get(int(icpsr), {}).get('party', 0) for icpsr in member_list
    ])

    defection_rates = np.zeros(n_members)
    valid_mask = ~np.isnan(vote_matrix)

    for party_code in [100, 200]:
        party_mask = party_labels == party_code
        if party_mask.sum() < 2:
            continue

        party_indices = np.where(party_mask)[0]
        party_votes = vote_matrix[party_mask]
        party_valid = valid_mask[party_mask]

        for idx in party_indices:
            others_mask = party_mask.copy()
            others_mask[idx] = False
            others_votes = vote_matrix[others_mask]
            others_valid = valid_mask[others_mask]

            my_valid = valid_mask[idx]
            both_valid = my_valid & (others_valid.sum(axis=0) >= 5)

            if both_valid.sum() == 0:
                continue

            my_votes = vote_matrix[idx, both_valid]
            others_v = others_votes[:, both_valid]
            others_majority = (np.nanmean(others_v, axis=0) > 0.5).astype(float)
            my_v = (my_votes > 0.5).astype(float)

            valid_both = ~np.isnan(my_votes)
            if valid_both.sum() == 0:
                continue

            defections = (my_v[valid_both] != others_majority[valid_both]).sum()
            defection_rates[idx] = defections / valid_both.sum()

    labels = (defection_rates >= threshold_pct).astype(int)
    return defection_rates, labels


def build_node_features(member_list, member_info, agreement, vote_matrix):
    n = len(member_list)
    valid_mask = ~np.isnan(vote_matrix)
    participation = valid_mask.sum(axis=1) / vote_matrix.shape[1]
    yea_rates = np.nanmean(np.where(valid_mask, vote_matrix, np.nan), axis=1)
    yea_rates = np.nan_to_num(yea_rates, nan=0.5)

    mean_agreement = agreement.mean(axis=1)

    parties = np.array([
        1 if member_info.get(int(icpsr), {}).get('party') == 200 else 0
        for icpsr in member_list
    ])

    features = np.zeros((n, 8))
    for i, icpsr in enumerate(member_list):
        info = member_info.get(int(icpsr), {})
        features[i, 0] = info.get('nominate_dim1', 0.0)
        features[i, 1] = info.get('nominate_dim2', 0.0)
        features[i, 2] = parties[i]
        features[i, 3] = participation[i]
        features[i, 4] = yea_rates[i]
        features[i, 5] = mean_agreement[i]

        same = (parties == parties[i]) & (np.arange(n) != i)
        diff = (parties != parties[i])
        features[i, 6] = agreement[i, diff].mean() if diff.sum() > 0 else 0.0
        features[i, 7] = agreement[i, same].mean() if same.sum() > 0 else 0.0

    return features


def process_all_congresses():
    all_results = {}

    for congress in CONGRESSES:
        existing = os.path.join(RESULTS_DIR, f"congress_{congress}.npz")
        if os.path.exists(existing) and os.path.exists(os.path.join(RESULTS_DIR, f"member_info_{congress}.json")):
            print(f"Congress {congress}: already processed, loading...")
            with open(os.path.join(RESULTS_DIR, f"member_info_{congress}.json")) as f:
                member_info = json.load(f)
            data = np.load(existing, allow_pickle=True)
            member_list = data['member_list']

            spectral = compute_spectral_features(data['agreement'])
            polarization = compute_polarization_metrics(
                {int(k): v for k, v in member_info.items()}, member_list
            )
            all_results[congress] = {
                'congress': congress,
                'n_members': len(member_list),
                'spectral': spectral,
                'polarization': polarization,
            }
            continue

        print(f"\nProcessing Congress {congress}...")
        votes, members = load_congress(congress)
        if votes is None:
            print(f"  Skipping: data not found")
            continue

        try:
            agreement, shared, member_list, member_info, vote_matrix = build_agreement_graph(votes, members)
        except Exception as e:
            print(f"  Error: {e}")
            continue

        print(f"  Members: {len(member_list)}, Rollcalls: {vote_matrix.shape[1]}")

        spectral = compute_spectral_features(agreement)
        print(f"  Fiedler: {spectral['fiedler']:.4f}")

        polarization = compute_polarization_metrics(member_info, member_list)
        print(f"  Party distance: {polarization['party_distance']:.4f}")

        node_features = build_node_features(member_list, member_info, agreement, vote_matrix)

        defection_rates_10, defection_labels_10 = compute_defection_labels(
            vote_matrix, member_list, member_info, 0.10
        )
        print(f"  Defectors@10%: {defection_labels_10.sum()}/{len(defection_labels_10)}")

        np.savez_compressed(
            os.path.join(RESULTS_DIR, f"congress_{congress}.npz"),
            agreement=agreement,
            vote_matrix=vote_matrix,
            node_features=node_features,
            member_list=np.array(member_list),
        )
        np.save(os.path.join(RESULTS_DIR, f"defection_rates_{congress}.npy"), defection_rates_10)
        np.save(os.path.join(RESULTS_DIR, f"defection_labels_{congress}.npy"), defection_labels_10)

        with open(os.path.join(RESULTS_DIR, f"member_info_{congress}.json"), 'w') as f:
            json.dump({str(k): v for k, v in member_info.items()}, f)

        all_results[congress] = {
            'congress': congress,
            'n_members': len(member_list),
            'n_rollcalls': int(vote_matrix.shape[1]),
            'spectral': spectral,
            'polarization': polarization,
        }

    with open(os.path.join(RESULTS_DIR, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nProcessed {len(all_results)} congresses")
    return all_results


if __name__ == "__main__":
    results = process_all_congresses()
