#!/usr/bin/env python3
import os, json, sys
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser("~/CongressionalGNN/data_senate")
RESULTS_DIR = os.path.expanduser("~/CongressionalGNN/senate_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

CONGRESSES = list(range(100, 119))


def process_senate(congress_num):
    votes_path = os.path.join(DATA_DIR, f"S{congress_num}_votes.csv")
    members_path = os.path.join(DATA_DIR, f"S{congress_num}_members.csv")

    if not os.path.exists(votes_path) or not os.path.exists(members_path):
        print(f"  Skipping S{congress_num}: data not found", flush=True)
        return None

    votes = pd.read_csv(votes_path, low_memory=False)
    members = pd.read_csv(members_path, low_memory=False)

    senate_votes = votes[votes['chamber'] == 'Senate'].copy()
    senate_votes = senate_votes[senate_votes['cast_code'].isin([1, 2, 3, 4, 5, 6])].copy()
    senate_votes['yea'] = senate_votes['cast_code'].isin([1, 2, 3]).astype(int)

    senate_members = members[members['chamber'] == 'Senate'].copy()
    senate_members = senate_members[senate_members['party_code'].isin([100, 200])].copy()

    valid_icpsrs = set(senate_members['icpsr'].unique())
    senate_votes = senate_votes[senate_votes['icpsr'].isin(valid_icpsrs)]

    vote_counts = senate_votes.groupby('icpsr').size()
    active = vote_counts[vote_counts >= 20].index.tolist()
    senate_votes = senate_votes[senate_votes['icpsr'].isin(active)]

    member_list = sorted(senate_votes['icpsr'].unique())
    member_to_idx = {m: i for i, m in enumerate(member_list)}
    n = len(member_list)

    if n < 10:
        print(f"  Skipping S{congress_num}: only {n} members", flush=True)
        return None

    rollcalls = sorted(senate_votes['rollnumber'].unique())
    roll_to_idx = {r: i for i, r in enumerate(rollcalls)}
    n_rolls = len(rollcalls)

    vote_matrix = np.full((n, n_rolls), np.nan)
    for _, row in senate_votes.iterrows():
        i = member_to_idx.get(row['icpsr'])
        j = roll_to_idx.get(row['rollnumber'])
        if i is not None and j is not None:
            vote_matrix[i, j] = row['yea']

    valid_mask = ~np.isnan(vote_matrix)
    both_valid = valid_mask.astype(np.float32) @ valid_mask.astype(np.float32).T

    vm_binary = np.where(valid_mask, vote_matrix, 0.0).astype(np.float32)
    both_yea = vm_binary @ vm_binary.T
    both_nay = (1.0 - vm_binary) * valid_mask.astype(np.float32)
    both_nay = both_nay @ both_nay.T

    agree_count = both_yea + both_nay
    agreement = np.divide(agree_count, both_valid, out=np.zeros_like(agree_count), where=both_valid >= 20)
    agreement[both_valid < 20] = 0.0
    np.fill_diagonal(agreement, 0.0)

    adj = (agreement > 0.5).astype(float)
    np.fill_diagonal(adj, 0)
    degrees = adj.sum(axis=1)
    valid = degrees > 0
    adj_clean = adj[np.ix_(valid, valid)]
    degrees_clean = adj_clean.sum(axis=1)
    n_clean = adj_clean.shape[0]

    if n_clean < 3:
        fiedler = 0.0
    else:
        D_inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(degrees_clean, 1e-10)))
        L_norm = np.eye(n_clean) - D_inv_sqrt @ adj_clean @ D_inv_sqrt
        try:
            eigenvalues = np.sort(np.real(np.linalg.eigvalsh(L_norm)))
            fiedler = float(eigenvalues[1]) if len(eigenvalues) > 1 else 0.0
        except:
            fiedler = 0.0

    dem_scores = []
    rep_scores = []
    members_indexed = senate_members.set_index('icpsr')
    for ic in member_list:
        if ic in members_indexed.index:
            row = members_indexed.loc[ic]
            if isinstance(row, pd.DataFrame):
                row = row.iloc[0]
            nom = row.get('nominate_dim1', 0)
            if pd.notna(nom):
                party = row.get('party_code', 0)
                if party == 100:
                    dem_scores.append(nom)
                elif party == 200:
                    rep_scores.append(nom)

    party_distance = abs(np.mean(rep_scores) - np.mean(dem_scores)) if dem_scores and rep_scores else 0

    density = float(adj.sum() / (n * (n - 1))) if n > 1 else 0
    mean_degree = float(degrees.mean())

    result = {
        'congress': congress_num,
        'chamber': 'Senate',
        'n_members': n,
        'n_rollcalls': n_rolls,
        'fiedler': fiedler,
        'party_distance': party_distance,
        'density': density,
        'mean_degree': mean_degree,
    }

    print(f"  S{congress_num}: {n} members, {n_rolls} votes, Fiedler={fiedler:.4f}, PartyDist={party_distance:.3f}", flush=True)
    return result


if __name__ == "__main__":
    all_results = {}
    for congress in CONGRESSES:
        print(f"\nProcessing Senate {congress}...", flush=True)
        result = process_senate(congress)
        if result:
            all_results[congress] = result

    with open(os.path.join(RESULTS_DIR, "senate_spectral.json"), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n\n=== SENATE SPECTRAL SUMMARY ===")
    print(f"{'Congress':<10} {'Years':<12} {'Members':>8} {'Fiedler':>10} {'Party Dist':>12} {'Density':>10}")
    print("-" * 65)

    years = {100: '1987-89', 101: '1989-91', 102: '1991-93', 103: '1993-95',
             104: '1995-97', 105: '1997-99', 106: '1999-01', 107: '2001-03',
             108: '2003-05', 109: '2005-07', 110: '2007-09', 111: '2009-11',
             112: '2011-13', 113: '2013-15', 114: '2015-17', 115: '2017-19',
             116: '2019-21', 117: '2021-23', 118: '2023-25'}

    for c in sorted(all_results.keys()):
        r = all_results[c]
        y = years.get(c, '???')
        print(f"{c:<10} {y:<12} {r['n_members']:>8} {r['fiedler']:>10.4f} {r['party_distance']:>12.3f} {r['density']:>10.3f}")

    print(f"\nResults saved to senate_results/senate_spectral.json")
