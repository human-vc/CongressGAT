# CongressGAT: Graph Attention Networks for Predicting Legislative Defection

This repository contains code and data for **"The Geometry of Partisanship: Graph Attention Networks for Predicting Legislative Defection in the U.S. House, 1995-2024."**

## Overview

We construct temporal voting networks for every session of the U.S. House from the 104th through 118th Congress (1995-2024) and apply Graph Attention Networks (GATs) to predict which members will defect from their party line.

**Key results:**
- GAT achieves F1=0.75 and AUC=0.88 at 5% defection threshold, outperforming logistic regression (F1=0.60), random forests (F1=0.63), and naive baselines (F1=0.59)
- Fiedler vector of voting networks correlates with DW-NOMINATE at r>0.97
- Attention mechanism reveals same-party connections receive 200x more weight than cross-party ties
- Tea Party wave of 2010 produced a permanent 40% decline in cross-party agreement

## Repository Structure

```
CongressionalGNN/
    pipeline_full.py          - Data processing, graph construction, spectral analysis
    model_gat.py              - GAT model, training, baselines, attention analysis
    generate_final_figures.py - Publication-quality figure generation
    data/                     - Voteview CSV files (members only; votes excluded for size)
    results_final/            - Experiment results (JSON)
    figures_final/            - Figures (PDF and PNG)
    paper_final/              - LaTeX source and compiled PDF
```

## Data

Roll-call data from [Voteview](https://voteview.com/) (Lewis et al., 2023). Member CSVs are included; vote CSVs are excluded due to size (~500MB total). Download them from:
- `https://voteview.com/static/data/out/votes/H{num}_votes.csv`
- `https://voteview.com/static/data/out/members/H{num}_members.csv`

## Requirements

- Python 3.10+
- PyTorch 2.x
- PyTorch Geometric 2.x
- pandas, numpy, scipy, scikit-learn, matplotlib

## Usage

```bash
python pipeline_full.py           - Process all congresses
python model_gat.py               - Train GAT and baselines
python generate_final_figures.py  - Generate figures
```

## Citation

If you use this code or data, please cite the paper.
