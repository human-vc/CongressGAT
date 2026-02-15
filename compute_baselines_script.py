import os
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# Add pipeline path to sys.path
sys.path.append(os.path.expanduser('~/CongressionalGNN'))
from congress_gat_pipeline import load_congress_data, compute_agreement_matrix, extract_node_features, compute_defection_labels

TRAIN_CONGRESSES = [110, 111, 112, 113, 114]
TEST_CONGRESSES = [115, 116, 117]

X_train, y_train = [], []
X_test_dict, y_test_dict = {}, {}

for c in TRAIN_CONGRESSES + TEST_CONGRESSES:
    try:
        members, votes = load_congress_data(c)
        agreement, icpsr_list, _ = compute_agreement_matrix(members, votes)
        features, _, _ = extract_node_features(members, icpsr_list)
        labels, _ = compute_defection_labels(members, votes, icpsr_list)
        
        if c in TRAIN_CONGRESSES:
            X_train.extend(features)
            y_train.extend(labels)
        else:
            X_test_dict[c] = features
            y_test_dict[c] = labels
            
    except Exception as e:
        print(f"Skipped {c}: {e}")

# Train RF
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Evaluate
print("Per-Congress Baseline AUCs:")
for c in TEST_CONGRESSES:
    if c in X_test_dict:
        probs = rf.predict_proba(X_test_dict[c])[:, 1]
        auc = roc_auc_score(y_test_dict[c], probs)
        print(f"Congress {c}: {auc:.3f}")
    else:
        print(f"Congress {c}: Missing")
