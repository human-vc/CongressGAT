#!/usr/bin/env python3
"""Run pipeline, save results, skip slow visualizations."""
import sys
sys.stdout.reconfigure(line_buffering=True)

from congress_gat_pipeline import (
    process_all_congresses, train_congress_gat, evaluate_model,
    run_baselines, attention_analysis, causal_analysis,
    CONGRESSES, TRAIN_CONGRESSES, TEST_CONGRESSES, OUTPUT_DIR
)
import json
import os
import torch

print("=" * 60)
print("CongressGAT Pipeline (fast mode)")
print("=" * 60)

# Step 1: Process all congresses
print("\n[1/6] Processing congressional data...", flush=True)
all_data = process_all_congresses()

# Step 2: Train model
print("\n[2/6] Training CongressGAT model...", flush=True)
model, train_losses = train_congress_gat(all_data, n_epochs=300, lr=0.005)

# Step 3: Evaluate
print("\n[3/6] Evaluating model...", flush=True)
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
print("\n[4/6] Running baselines...", flush=True)
baseline_results = run_baselines(all_data)
print(f"  LR test coalition F1: {baseline_results['logistic_regression']['test']['coalition_f1']:.4f}")
print(f"  RF test coalition F1: {baseline_results['random_forest']['test']['coalition_f1']:.4f}")
print(f"  LR test defection AUC: {baseline_results['logistic_regression']['test']['defection_auc']:.4f}")
print(f"  RF test defection AUC: {baseline_results['random_forest']['test']['defection_auc']:.4f}")

# Step 5: Attention analysis
print("\n[5/6] Analyzing attention weights...", flush=True)
attention_results = attention_analysis(model, all_data, metrics)
for cnum in CONGRESSES:
    if cnum in attention_results:
        ar = attention_results[cnum]
        print(f"  Congress {cnum}: same={ar['mean_same_party_attention']:.5f}, "
              f"cross={ar['mean_cross_party_attention']:.5f}, "
              f"ratio={ar['mean_same_party_attention']/max(ar['mean_cross_party_attention'],1e-8):.3f}")

# Step 6: Causal analysis
print("\n[6/6] Running causal analysis...", flush=True)
causal_results = causal_analysis(all_data)
print(f"  Tea Party effect: {causal_results['tea_party']['estimated_effect']:.4f}")
print(f"  Trump effect: {causal_results['trump_era']['estimated_effect']:.4f}")

# Per-congress details
print("\n  Per-congress results:")
for c in CONGRESSES:
    pc = metrics['per_congress'][c]
    split = 'TRAIN' if c in TRAIN_CONGRESSES else 'TEST'
    print(f"    {c}th [{split}]: pol={pc['polarization_true']:.3f}, "
          f"coal_f1={pc['coalition_f1']:.3f}, def_auc={pc['defection_auc']:.3f}")

# Save all results
results_summary = {
    'train_metrics': metrics['train'],
    'test_metrics': metrics['test'],
    'per_congress': {str(k): v for k, v in metrics['per_congress'].items()},
    'predictions': metrics['predictions'],
    'baselines': baseline_results,
    'causal': causal_results,
    'attention_summary': {
        str(cnum): {k: v for k, v in attention_results[cnum].items() if k != 'attention_matrix'}
        for cnum in CONGRESSES if cnum in attention_results
    },
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, 'full_results.json'), 'w') as f:
    json.dump(results_summary, f, indent=2, default=str)

torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, 'congress_gat_model.pt'))

# Also save data summaries for the paper
data_summary = {}
for cnum in CONGRESSES:
    d = all_data[cnum]
    valid = d['party_labels'] < 2
    data_summary[str(cnum)] = {
        'n_members': len(d['icpsr_list']),
        'n_edges': int(d['adj_binary'].sum()) // 2,
        'polarization': float(d['polarization']),
        'within_agreement': float(d['within_avg']),
        'between_agreement': float(d['between_avg']),
        'alignment': float(d['alignment']),
        'n_defectors': int(d['defection_labels'].sum()),
        'defection_rate': float(d['defection_labels'].mean()),
        'n_democrats': int((d['party_labels'] == 0).sum()),
        'n_republicans': int((d['party_labels'] == 1).sum()),
        'mean_nom1_dem': float(d['features'][d['party_labels'] == 0, 0].mean()),
        'mean_nom1_rep': float(d['features'][d['party_labels'] == 1, 0].mean()),
    }

with open(os.path.join(OUTPUT_DIR, 'data_summary.json'), 'w') as f:
    json.dump(data_summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/")
print("Pipeline complete!")
