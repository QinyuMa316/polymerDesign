
## Code for: Multi-Objective Inverse Design of High-Performance Polymers via Contrastive-Pretrained Graph Attention Networks and NSGA-II


## Requirements
```
conda create -n polydesign --file requirements.txt
conda activate polydesign
```

## Notation

- Tg denotes the glass transition temperature $T_g$.
- EPS (epsilon) denotes the dielectric constant $k$ (also written as $\varepsilon$).

## Data

The datasets used in this project are publicly available at:

https://doi.org/10.6084/m9.figshare.31493752

Alternatively, you can access them directly from the polyDesign_data directory in this repository.

The data directory is organized as follows:

```
polyDesign_data
├── data_modification
│   └── polyimides_pred_prop_base_str_std.csv
└── data_train
    ├── PI1M.csv
    ├── EPS_combined.csv
    └── Tg_combined.csv
```

The files are used as follows:

- `PI1M.csv` is used for model pretraining.
- `EPS_combined.csv` and `Tg_combined.csv` are used for fine-tuning on the $k$ and $T_g$ prediction tasks, respectively.
- `polyimides_pred_prop_base_str_std.csv` is used for polymer structure optimization.

## Usage

Run the pretraining script with the desired model configuration:

### Pretraining
```
python3 polymerCL-GNN-Pretrain-mix.py \
    --n_layers 3 \
    --hidden_dim 128 \
    --dropout 0.02 \
    --lr_pre 1e-4 \
    --epoch_pre 50 \
    --batch_size_pre 128 \
    --model_dir models_polymers
```
Main arguments:
+ `--n_layers`: Number of graph neural network layers.
+ `--hidden_dim`: Dimension of the hidden representations.
+ `--dropout`: Dropout rate.
+ `--lr_pre`: Learning rate for pretraining.
+ `--epoch_pre`: Number of pretraining epochs.
+ `--batch_size_pre`: Batch size used during pretraining.
+ `--model_dir`: Directory used to save pretrained model checkpoints.

### Fine-Tuning
After pretraining, run the fine-tuning script for the target property:
```
python3 polymerCL-GNN-FT.py \
    --pretrain_epoch 50 \
    --model_dir models_polymers \
    --target EPS \
    --n_layers 3 \
    --hidden_dim 128 \
    --dropout 0.02 \
    --lr 1e-5 \
    --batch_size 4 \
    --n_enum 12
```
Main arguments:
+ `--pretrain_epoch`: Pretraining checkpoint epoch to load.
+ `--model_dir`: Directory containing the pretrained model checkpoints.
+ `--target`: Target property for fine-tuning. Supported values are EPS and Tg.
+ `--n_layers`: Number of graph neural network layers.
+ `--hidden_dim`: Dimension of the hidden representations.
+ `--dropout`: Dropout rate.
+ `--lr`: Learning rate for fine-tuning.
+ `--batch_size`: Batch size used during fine-tuning.
+ `--n_enum`: Number of molecular enumerations used for data augmentation.

### Functional Group Attribution Analysis

```
python FgAttrAnalysisSDEmultipro.py
```

### Structure Optimization

```
python FgAttrAnalysisSDEmultipro.py
```

## Computational Resources

The experiments were conducted using an NVIDIA A100 GPU.

## Notes

Results may vary slightly across runs due to differences in random initialization, data partitioning, hyperparameter settings, and hardware-dependent numerical behavior.


