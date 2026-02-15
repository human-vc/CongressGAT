# CongressGAT: Tracking Congressional Polarization with Graph Attention Networks

A comprehensive analysis of U.S. congressional polarization using graph neural networks, spanning 19 Congresses (100th through 118th, 1987-2025).

## Overview

This repository contains the code, data processing pipeline, and paper source for a study that:

1. Constructs co-voting agreement networks for every Congress from the 100th through the 118th
2. Performs spectral analysis revealing a 94%+ decline in network connectivity over three decades
3. Trains a Graph Attention Network with temporal attention for three prediction tasks:
   - **Defection forecasting** (AUC: 0.908 on held-out Congresses)
   - **Coalition detection** (F1 > 0.97)
   - **Polarization trajectory prediction**
4. Identifies the Tea Party wave of 2010 as the single largest structural shock to congressional cooperation

## Key Finding

The Fiedler value (algebraic connectivity) of the House co-voting network has collapsed from 0.534 in 1987 to 0.032 in 2023. The Tea Party wave alone accounts for roughly 87% of this decline.

## Repository Structure

```
CongressGAT/
├── data/                    # Voteview roll-call data (H100-H118)
├── pipeline_v2.py           # Data processing and graph construction
├── model.py                 # GAT architecture and training
├── generate_all_figures.py  # Publication-quality figure generation
├── paper/
│   ├── main.tex            # Paper source
│   └── references.bib      # Bibliography
├── paper_figures/           # Generated figures (PDF and PNG)
├── pipeline_results/        # Intermediate data (agreement matrices, features)
└── model_results/           # Trained model and evaluation results
```

## Data

Roll-call voting records from [Voteview](https://voteview.com/) (Lewis et al., 2023). Each Congress includes:
- Vote records for all House roll calls
- Member information including party affiliation and DW-NOMINATE scores

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Geometric
- NumPy, Pandas, SciPy, scikit-learn
- Matplotlib (for figures)

## Usage

```bash
python pipeline_v2.py          # Process all Congresses
python model.py                # Train GAT and run baselines
python generate_all_figures.py # Generate figures
```

## Paper

The paper can be compiled with pdflatex:

```bash
cd paper
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Citation

If you use this code or data, please cite:

```bibtex
@article{congressgat2025,
    title={The Geometry of Gridlock: Tracking Congressional Polarization with Graph Attention Networks},
    year={2025}
}
```

## License

MIT
