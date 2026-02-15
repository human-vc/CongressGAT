import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.stats import pearsonr, spearmanr
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

DATA_DIR = os.path.expanduser('~/CongressionalGNN/data')
RESULTS_DIR = os.path.expanduser('~/CongressionalGNN/results_final')
os.makedirs(RESULTS_DIR, exist_ok=True)

CONGRESSES = list(range(104, 119))

def load_congress_data(congress_num):
    votes_path = os.path.join(DATA_DIR, f'H{congress_num}_votes.csv')
    members_path = os.path.join(DATA_DIR, f'H{congress_num}_members.csv')
    votes = pd.read_csv(votes_path)
    members = pd.read_csv(members_path)
    members = members[members['chamber'] == 'House'].copy()
    members = members[members['party_code'].isin([100, 200])].copy()
    votes = votes[votes['chamber'] == 'House'].copy()
    votes = votes[votes['icpsr'].isin(members['icpsr'].values)].copy()
    return votes, members

def build_agreement_matrix_fast(votes, members):
    member_ids = sorted(members['icpsr'].unique())
    id_to_idx = {mid: i for i, mid in enumerate(member_ids)}
    n = len(member_ids)

    votes_filtered = votes[votes['cast_code'].isin([1, 2, 3, 4, 5, 6])].copy()
    votes_filtered['yea'] = votes_filtered['cast_code'].isin([1, 2, 3]).astype(np.int8)
    votes_filtered = votes_filtered[votes_filtered['icpsr'].isin(set(member_ids))]
    votes_filtered['idx'] = votes_filtered['icpsr'].map(id_to_idx)

    pivot = votes_filtered.pivot_table(index='idx', columns='rollnumber', values='yea', fill_value=-1)
    vote_matrix = pivot.values
    member_order = pivot.index.values

    valid = (vote_matrix >= 0)
    yea_mat = np.where(valid, vote_matrix, 0).astype(np.float32)
    nay_mat = np.where(valid, 1 - vote_matrix, 0).astype(np.float32)

    agreement = yea_mat @ yea_mat.T + nay_mat @ nay_mat.T
    overlap = valid.astype(np.float32) @ valid.astype(np.float32).T

    np.fill_diagonal(overlap, 1)
    agreement_rate_sub = agreement / overlap
    np.fill_diagonal(agreement_rate_sub, 0)

    agreement_rate = np.zeros((n, n), dtype=np.float32)
    ix = np.ix_(member_order, member_order)
    agreement_rate[ix] = agreement_rate_sub

    min_votes = 10
    overlap_full = np.zeros((n, n), dtype=np.float32)
    overlap_full[ix] = overlap
    np.fill_diagonal(overlap_full, 0)
    agreement_rate[overlap_full < min_votes] = 0

    return agreement_rate, member_ids, id_to_idx

def spectral_analysis(agreement_rate, members, member_ids):
    n = agreement_rate.shape[0]
    adj = agreement_rate.copy()
    np.fill_diagonal(adj, 0)

    degree = adj.sum(axis=1)
    D_inv_sqrt = np.diag(np.where(degree > 0, 1.0 / np.sqrt(degree), 0))
    L = np.diag(degree) - adj
    L_norm = D_inv_sqrt @ L @ D_inv_sqrt

    eigenvalues, eigenvectors = np.linalg.eigh(L_norm)
    fiedler_vector = eigenvectors[:, 1]
    fiedler_value = eigenvalues[1]

    member_id_to_party = {}
    member_id_to_nominate = {}
    for _, row in members.iterrows():
        member_id_to_party[row['icpsr']] = row['party_code']
        member_id_to_nominate[row['icpsr']] = row.get('nominate_dim1', np.nan)

    parties = np.array([member_id_to_party.get(mid, 0) for mid in member_ids])
    nominates = np.array([member_id_to_nominate.get(mid, np.nan) for mid in member_ids])

    valid_mask = ~np.isnan(nominates)
    if valid_mask.sum() > 10:
        corr, pval = pearsonr(fiedler_vector[valid_mask], nominates[valid_mask])
    else:
        corr, pval = 0, 1

    return {
        'fiedler_value': float(fiedler_value),
        'fiedler_vector': fiedler_vector,
        'eigenvalues': eigenvalues[:20].tolist(),
        'fiedler_nominate_corr': float(corr),
        'fiedler_nominate_pval': float(pval),
        'parties': parties,
        'nominates': nominates
    }

def compute_party_cohesion(agreement_rate, members, member_ids):
    member_id_to_party = {}
    for _, row in members.iterrows():
        member_id_to_party[row['icpsr']] = row['party_code']

    parties = np.array([member_id_to_party.get(mid, 0) for mid in member_ids])
    dem_idx = np.where(parties == 100)[0]
    rep_idx = np.where(parties == 200)[0]

    dem_block = agreement_rate[np.ix_(dem_idx, dem_idx)]
    np.fill_diagonal(dem_block, np.nan)
    dem_cohesion = float(np.nanmean(dem_block)) if len(dem_idx) > 1 else 0

    rep_block = agreement_rate[np.ix_(rep_idx, rep_idx)]
    np.fill_diagonal(rep_block, np.nan)
    rep_cohesion = float(np.nanmean(rep_block)) if len(rep_idx) > 1 else 0

    cross_block = agreement_rate[np.ix_(dem_idx, rep_idx)]
    cross_agreement = float(np.mean(cross_block)) if len(dem_idx) > 0 and len(rep_idx) > 0 else 0

    return {
        'dem_cohesion': dem_cohesion,
        'rep_cohesion': rep_cohesion,
        'cross_party_agreement': cross_agreement,
        'n_dem': len(dem_idx),
        'n_rep': len(rep_idx)
    }

def compute_defection_rates(votes, members, member_ids):
    member_id_to_party = {}
    for _, row in members.iterrows():
        member_id_to_party[row['icpsr']] = row['party_code']

    vf = votes[votes['cast_code'].isin([1, 2, 3, 4, 5, 6])].copy()
    vf['yea'] = vf['cast_code'].isin([1, 2, 3]).astype(int)
    vf = vf[vf['icpsr'].isin(set(member_ids))]
    vf['party'] = vf['icpsr'].map(member_id_to_party)

    party_maj = vf.groupby(['rollnumber', 'party'])['yea'].mean().reset_index()
    party_maj['majority_vote'] = (party_maj['yea'] > 0.5).astype(int)
    party_maj_dict = {(r, p): m for r, p, m in zip(party_maj['rollnumber'], party_maj['party'], party_maj['majority_vote'])}

    vf['party_majority'] = vf.apply(lambda r: party_maj_dict.get((r['rollnumber'], r['party']), -1), axis=1)
    vf = vf[vf['party_majority'] >= 0]
    vf['defected'] = (vf['yea'] != vf['party_majority']).astype(int)

    member_stats = vf.groupby('icpsr').agg(
        total_votes=('defected', 'count'),
        defections=('defected', 'sum')
    ).reset_index()
    member_stats['defection_rate'] = member_stats['defections'] / member_stats['total_votes']

    defection_rates = dict(zip(member_stats['icpsr'], member_stats['defection_rate']))
    return defection_rates

def build_member_features(members, member_ids, defection_rates_prev=None):
    features = []
    for mid in member_ids:
        row = members[members['icpsr'] == mid].iloc[0]
        nom1 = row.get('nominate_dim1', 0)
        nom2 = row.get('nominate_dim2', 0)
        if pd.isna(nom1): nom1 = 0
        if pd.isna(nom2): nom2 = 0
        party = 1 if row['party_code'] == 200 else 0
        seniority = row.get('nominate_number_of_votes', 0)
        if pd.isna(seniority): seniority = 0
        seniority = seniority / 1500.0
        prev_def = defection_rates_prev.get(mid, 0) if defection_rates_prev else 0
        features.append([nom1, nom2, party, seniority, prev_def, abs(nom1)])
    return np.array(features, dtype=np.float32)

def process_all_congresses():
    all_results = {}
    all_agreement_matrices = {}
    all_member_ids = {}
    all_members_data = {}
    all_features = {}
    all_defection_rates = {}
    all_spectral_data = {}

    prev_defection = None

    for cong in CONGRESSES:
        print(f"\n{'='*60}")
        print(f"Congress {cong}")
        print(f"{'='*60}")
        votes, members = load_congress_data(cong)
        print(f"  Members: {len(members)} | Vote records: {len(votes):,}")

        agreement_rate, member_ids, id_to_idx = build_agreement_matrix_fast(votes, members)
        print(f"  Agreement matrix: {agreement_rate.shape[0]}x{agreement_rate.shape[1]}")

        spectral = spectral_analysis(agreement_rate, members, member_ids)
        print(f"  Fiedler value: {spectral['fiedler_value']:.6f}")
        print(f"  Fiedler-NOMINATE r={spectral['fiedler_nominate_corr']:.4f} (p={spectral['fiedler_nominate_pval']:.2e})")

        cohesion = compute_party_cohesion(agreement_rate, members, member_ids)
        print(f"  Dem cohesion: {cohesion['dem_cohesion']:.4f} | Rep cohesion: {cohesion['rep_cohesion']:.4f}")
        print(f"  Cross-party agreement: {cohesion['cross_party_agreement']:.4f}")

        defection_rates = compute_defection_rates(votes, members, member_ids)
        defection_labels = {mid: (1 if defection_rates.get(mid, 0) > 0.10 else 0) for mid in member_ids}
        n_defectors = sum(defection_labels.values())
        mean_defection = np.mean(list(defection_rates.values()))
        print(f"  Defectors (>10%): {n_defectors}/{len(member_ids)} ({100*n_defectors/len(member_ids):.1f}%)")
        print(f"  Mean defection rate: {mean_defection:.4f}")

        features = build_member_features(members, member_ids, prev_defection)

        all_agreement_matrices[cong] = agreement_rate
        all_member_ids[cong] = member_ids
        all_members_data[cong] = members
        all_features[cong] = features
        all_defection_rates[cong] = defection_rates
        all_spectral_data[cong] = spectral

        spectral_save = {k: v for k, v in spectral.items() if k not in ['fiedler_vector', 'parties', 'nominates']}
        all_results[cong] = {
            'spectral': spectral_save,
            'cohesion': cohesion,
            'n_members': len(member_ids),
            'n_defectors_10pct': n_defectors,
            'mean_defection_rate': float(mean_defection),
        }

        prev_defection = defection_rates

    with open(os.path.join(RESULTS_DIR, 'congress_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    np.savez(os.path.join(RESULTS_DIR, 'processed_data.npz'),
             congresses=CONGRESSES,
             **{f'agreement_{c}': all_agreement_matrices[c] for c in CONGRESSES},
             **{f'features_{c}': all_features[c] for c in CONGRESSES},
             **{f'member_ids_{c}': all_member_ids[c] for c in CONGRESSES})

    for cong in CONGRESSES:
        np.save(os.path.join(RESULTS_DIR, f'defection_rates_{cong}.npy'),
                np.array([(mid, all_defection_rates[cong].get(mid, 0)) for mid in all_member_ids[cong]]))

    print(f"\n\nAll results saved to {RESULTS_DIR}")
    return all_results, all_agreement_matrices, all_member_ids, all_members_data, all_features, all_defection_rates, all_spectral_data

if __name__ == '__main__':
    process_all_congresses()
