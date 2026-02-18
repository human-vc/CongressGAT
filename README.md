# The Geometry of Gridlock: Tracking Congressional Polarization with Graph Attention Networks

**February 2026**

## Summary

Congressional polarization is typically measured by placing legislators on an ideological spectrum, but this approach misses the relational structure of legislative cooperation. We ask whether network topology can detect polarization dynamics that standard ideological measures overlook.

We construct co-voting networks for every U.S. House Congress from the 100th (1987) through the 118th (2025), connecting legislators who agree on a majority of shared roll-call votes, and track their spectral properties over time.

### Key Findings

- **Structural Collapse:** The algebraic connectivity of the House (Fiedler value) collapsed from a peak of **0.843** in the 107th Congress (post-9/11) to **0.032** in the 118th, indicating a near-complete structural disconnection between partisan blocs.
- **Predictive Power:** A Graph Attention Network (GAT) augmented with temporal attention achieves a mean AUC of **0.908** for predicting individual defection from party lines on held-out Congresses (115th–117th).
- **The Tea Party Shock:** An interrupted time series analysis identifies the 2010 Tea Party wave as the single largest structural shock in the dataset, permanently impairing the network's capacity for self-repair.
- **2023 Speaker Crisis:** The model identified **12 of 20** Republican holdouts in the McCarthy Speaker crisis from network position alone, including members whose ideological scores gave no indication of rebellion. Standard ideological distance would have flagged only 3.
- **Forecast:** The model projects continued decline for the 119th Congress (Fiedler 0.028), predicting persistent structural gridlock.

## Methodology

- **Data:** Roll-call votes from Voteview (100th–118th Congress).
- **Network Construction:** Edges represent >50% agreement on shared votes.
- **Model:** Temporal Graph Attention Network (GAT) learning representations of legislative behavior from local network topology and historical evolution.
- **Metrics:** Spectral algebraic connectivity (Fiedler value), modularity, and a novel Structural Resilience Index.

## Repository Structure

```
data/           Network edge lists and node features
models/         PyTorch GAT implementation
analysis/       Spectral analysis and time series scripts
paper/          LaTeX source and PDF
```

## License

MIT License. Copyright (c) 2026 Jacob Crainic.
