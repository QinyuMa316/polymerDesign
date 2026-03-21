
# Code for: Multi-Objective Inverse Design of High-Performance Polymers via Contrastive-Pretrained Graph Attention Networks and NSGA-II
 

## Requirements
```
conda create -n polydesign python=3.11
conda activate polydesign
conda install -y -c pytorch -c pyg -c conda-forge \
  numpy pandas matplotlib tqdm scikit-learn rdkit \
  pytorch cpuonly torch-geometric \
  seaborn umap-learn

conda install -y "numpy<2"
```

## Notation

- Tg denotes the glass transition temperature $T_g$.
- EPS (epsilon) denotes the dielectric constant $k$ (also written as $\varepsilon$).

## Data

The training data is in 10.6084/m9.figshare.31493752

