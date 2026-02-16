# CongressGAT: Graph Attention Networks for Predicting Legislative Defection

Code and data for **"The Geometry of Partisanship: Graph Attention Networks for Predicting Legislative Defection in the U.S. House, 1995–2024"** — submitted to [IC2S2 2026](https://ic2s2.org/).

## Overview

We construct temporal voting networks for every session of the U.S. House from the 104th through 118th Congress (1995–2024) and apply Graph Attention Networks (GATs) to predict which members will defect from their party line.

**Key results:**
- GAT achieves **F1 = 0.75** and **AUC = 0.88** at a 5% defection threshold, outperforming logistic regression (F1 = 0.60), random forests (F1 = 0.63), and naïve baselines (F1 = 0.59)
- Fiedler vector of voting networks correlates with DW-NOMINATE at *r* > 0.97
- Attention mechanism reveals same-party connections receive 200× more weight than cross-party ties
- The Tea Party wave of 2010 produced a permanent 40% decline in cross-party agreement

## Repository Structure

```
CongressGAT/
├── pipeline_fast.py           # Data processing, graph construction, spectral analysis
├── model_gat.py               # GAT model, training, baselines, attention analysis
├── data/                      # Voteview member CSVs (H100–H118)
├── pipeline_results/          # Processed pipeline outputs (JSON)
├── model_results/             # Model experiment results
├── results_final/             # Final experiment results
├── figures/                   # Exploratory figures
├── figures_final/             # Publication-quality figures (PDF/PNG)
├── paper/                     # Paper drafts (LaTeX)
├── paper_final/               # Final paper source
├── ic2s2_submission/          # IC2S2 2026 abstract and figures
├── generate_final_figures.py  # Publication-quality figure generation
├── generate_all_figures.py    # Comprehensive figure generation
├── generate_figures.py        # Figure generation utilities
├── build_graph.py             # Graph construction utilities
├── run_ablation.py            # Ablation study runner
├── compute_baselines_script.py# Baseline computation
├── senate_spectral.py         # Senate spectral analysis
└── legacy/                    # Archived earlier pipeline scripts
```

## Data

Roll-call data from [Voteview](https://voteview.com/) (Lewis et al., 2023). Member CSVs are included in `data/`; vote CSVs are excluded due to size (~500 MB total). Download them from:

```
https://voteview.com/static/data/out/votes/H{num}_votes.csv
```

Place downloaded vote CSVs in `data/`.

## Requirements

- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric 2.x
- pandas, numpy, scipy, scikit-learn, matplotlib

## Usage

```bash
# Process all congresses (graph construction + spectral analysis)
python pipeline_fast.py

# Train GAT and baselines
python model_gat.py

# Generate publication figures
python generate_final_figures.py
```

## Spectral Analysis

Fiedler values are computed using binary thresholding (τ = 0.5) on pairwise agreement scores, with isolated-node removal and the normalized Laplacian (L = I − D⁻¹ᐟ²AD⁻¹ᐟ²). See Section 3.2 of the paper for details. Canonical results are in `pipeline_results/`.

## Citation

If you use this code or data, please cite:

> Crainic, J. (2026). The Geometry of Partisanship: Graph Attention Networks for Predicting Legislative Defection in the U.S. House, 1995–2024. *IC2S2 2026*.

## License

See [LICENSE](LICENSE).
