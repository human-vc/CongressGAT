#!/usr/bin/env python3

import os
import json
import numpy as np
import pandas as pd
import warnings
import time
warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "pipeline_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONGRESSES = list(range(100, 119))


def process_congress(congress_num):
    t0 = time.time()
    votes_path = os.path.join(DATA_DIR, f"H{congress_num}_votes.csv")
    members_path = os.path.join(DATA_DIR, f"H{congress_num}_members.csv")

    if not os.path.exists(votes_path) or not os.path.exists(members_path):
        return None

    votes = pd.read_csv(votes_path, low_memory=False)
    members = pd.read_csv(members_path, low_memory=False)

    house_votes = votes[(votes['chamber'] == 'House') & (votes['cast_code'].isin([1, 2, 3, 4, 5, 6]))].copy()
    house_votes['yea'] = house_votes['cast_code'].isin([1, 2, 3]).astype(int)

    house_members = members[(members['chamber'] == 'House') & (members['party_code'].isin([100, 200]))].copy()
    valid_icpsrs = set(house_members['icpsr'].unique())
    house_votes = house_votes[house_votes['icpsr'].isin(valid_icpsrs)]

    vote_counts = house_votes.groupby('icpsr').size()
    active = set(vote_counts[vote_counts >= 50].index)
    house_votes = house_votes[house_votes['icpsr'].isin(active)]

    member_list = sorted(house_votes['icpsr'].unique())
    member_to_idx = {m: i for i, m in enumerate(member_list)}
    n = len(member_list)

    rollcalls = sorted(house_votes['rollnumber'].unique())
    roll_to_idx = {r: i for i, r in enumerate(rollcalls)}
    n_rolls = len(rollcalls)

    house_votes['m_idx'] = house_votes['icpsr'].map(member_to_idx)
    house_votes['r_idx'] = house_votes['rollnumber'].map(roll_to_idx)
    house_votes = house_votes.dropna(subset=['m_idx', 'r_idx'])

    vote_matrix = np.full((n, n_rolls), np.nan, dtype=np.float32)
    vote_matrix[house_votes['m_idx'].values.astype(int), house_votes['r_idx'].values.astype(int)] = house_votes['yea'].values.astype(np.float32)

    print(f"  [{time.time()-t0:.1f}s] Built vote matrix: {n} members x {n_rolls} rollcalls")

    valid_mask = ~np.isnan(vote_matrix)
    vm_zero = np.where(valid_mask, vote_matrix, 0.0).astype(np.float32)
    vm_inv = np.where(valid_mask, 1.0 - vote_matrix, 0.0).astype(np.float32)
    valid_f = valid_mask.astype(np.float32)

    shared = valid_f @ valid_f.T
    agree = (vm_zero @ vm_zero.T) + (vm_inv @ vm_inv.T)

    min_shared = 20
    agreement = np.divide(agree, shared, out=np.zeros_like(agree), where=shared >= min_shared)
    agreement[shared < min_shared] = 0.0
    np.fill_diagonal(agreement, 0.0)

    print(f"  [{time.time()-t0:.1f}s] Agreement matrix computed")

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

    adj = (agreement > 0.5).astype(float)
    np.fill_diagonal(adj, 0)
    degrees = adj.sum(axis=1)
    valid_nodes = degrees > 0
    adj_c = adj[np.ix_(valid_nodes, valid_nodes)]
    deg_c = adj_c.sum(axis=1)
    n_c = adj_c.shape[0]

    fiedler = 0.0
    if n_c >= 3:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(deg_c, 1e-10)))
        L_norm = np.eye(n_c) - D_inv_sqrt @ adj_c @ D_inv_sqrt
        try:
            evals = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
            fiedler = float(evals[1]) if len(evals) > 1 else 0.0
        except:
            pass

    spectral = {
        'fiedler': fiedler,
        'n_nodes': int(n),
        'n_edges': int(adj.sum() / 2),
        'density': float(adj.sum() / (n * (n - 1))) if n > 1 else 0.0,
        'mean_degree': float(degrees.mean()),
    }

    dem_scores = [member_info[int(m)]['nominate_dim1'] for m in member_list if member_info.get(int(m), {}).get('party') == 100]
    rep_scores = [member_info[int(m)]['nominate_dim1'] for m in member_list if member_info.get(int(m), {}).get('party') == 200]

    polarization = {
        'party_distance': float(abs(np.mean(rep_scores) - np.mean(dem_scores))) if dem_scores and rep_scores else 0.0,
        'dem_mean': float(np.mean(dem_scores)) if dem_scores else 0.0,
        'rep_mean': float(np.mean(rep_scores)) if rep_scores else 0.0,
        'dem_std': float(np.std(dem_scores)) if len(dem_scores) > 1 else 0.0,
        'rep_std': float(np.std(rep_scores)) if len(rep_scores) > 1 else 0.0,
        'overlap': float(max(0, max(dem_scores) - min(rep_scores))) if dem_scores and rep_scores else 0.0,
        'n_dem': len(dem_scores),
        'n_rep': len(rep_scores),
    }

    parties = np.array([1 if member_info.get(int(m), {}).get('party') == 200 else 0 for m in member_list])
    participation = valid_mask.sum(axis=1) / n_rolls
    yea_rates = np.nanmean(np.where(valid_mask, vote_matrix, np.nan), axis=1)
    yea_rates = np.nan_to_num(yea_rates, nan=0.5)
    mean_agr = agreement.mean(axis=1)

    features = np.zeros((n, 8), dtype=np.float32)
    for i, icpsr in enumerate(member_list):
        info = member_info.get(int(icpsr), {})
        features[i, 0] = info.get('nominate_dim1', 0.0)
        features[i, 1] = info.get('nominate_dim2', 0.0)
        features[i, 2] = parties[i]
        features[i, 3] = participation[i]
        features[i, 4] = yea_rates[i]
        features[i, 5] = mean_agr[i]
        same = (parties == parties[i]) & (np.arange(n) != i)
        diff = (parties != parties[i])
        features[i, 6] = agreement[i, diff].mean() if diff.sum() > 0 else 0.0
        features[i, 7] = agreement[i, same].mean() if same.sum() > 0 else 0.0

    defection_rates = np.zeros(n, dtype=np.float32)
    for party_code in [100, 200]:
        party_mask = parties == (1 if party_code == 200 else 0)
        if party_mask.sum() < 2:
            continue
        party_indices = np.where(party_mask)[0]

        for idx in party_indices:
            others = party_mask.copy()
            others[idx] = False
            if others.sum() == 0:
                continue

            my_valid = valid_mask[idx]
            others_valid_count = valid_mask[others].sum(axis=0)
            usable = my_valid & (others_valid_count >= 5)

            if usable.sum() == 0:
                continue

            my_votes_u = vote_matrix[idx, usable]
            others_mean = np.nanmean(vote_matrix[others][:, usable], axis=0)
            others_majority = (others_mean > 0.5).astype(float)
            my_binary = (my_votes_u > 0.5).astype(float)

            not_nan = ~np.isnan(my_votes_u)
            if not_nan.sum() > 0:
                defection_rates[idx] = (my_binary[not_nan] != others_majority[not_nan]).sum() / not_nan.sum()

    defection_labels = (defection_rates >= 0.10).astype(int)

    print(f"  [{time.time()-t0:.1f}s] Fiedler={fiedler:.4f}, Distance={polarization['party_distance']:.3f}, Defectors={defection_labels.sum()}/{n}")

    np.savez_compressed(
        os.path.join(RESULTS_DIR, f"congress_{congress_num}.npz"),
        agreement=agreement,
        vote_matrix=vote_matrix,
        node_features=features,
        member_list=np.array(member_list),
    )
    np.save(os.path.join(RESULTS_DIR, f"defection_rates_{congress_num}.npy"), defection_rates)
    np.save(os.path.join(RESULTS_DIR, f"defection_labels_{congress_num}.npy"), defection_labels)
    with open(os.path.join(RESULTS_DIR, f"member_info_{congress_num}.json"), 'w') as f:
        json.dump({str(k): v for k, v in member_info.items()}, f)

    return {
        'congress': congress_num,
        'n_members': n,
        'n_rollcalls': n_rolls,
        'spectral': spectral,
        'polarization': polarization,
        'defection': {
            'n_defectors': int(defection_labels.sum()),
            'pct_defectors': float(defection_labels.mean()),
        }
    }


def main():
    all_results = {}
    for congress in CONGRESSES:
        print(f"\n=== Congress {congress} ===")
        result = process_congress(congress)
        if result:
            all_results[congress] = result

    with open(os.path.join(RESULTS_DIR, "all_results.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\nDone: {len(all_results)} congresses processed")
    for c in sorted(all_results.keys()):
        r = all_results[c]
        print(f"  H{c}: {r['n_members']} members, Fiedler={r['spectral']['fiedler']:.4f}, PartyDist={r['polarization']['party_distance']:.3f}")


if __name__ == "__main__":
    main()
