
## Code for: Multi-Objective Inverse Design of High-Performance Polymers via Contrastive-Pretrained Graph Attention Networks and NSGA-II


## Requirements
optional:
```
conda create -n polydesign python=3.11
conda activate polydesign
```
```
python -m pip install -r requirements.txt
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
├── process_data.py
├── data_modification
│   └── polyimides_pred_prop_base_str_std.csv
├── data_raw
|   ├── PI1M.csv
│   ├── EPS_combined.csv
│   └── Tg_combined.csv
└── data_train
    ├── PI1M_unique.csv
    ├── EPS_combined_unique.csv
    └── Tg_combined_unique.csv
```
The files are used as follows:

- `process_data.py` performs data preprocessing, including SMILES validity checking, standardization, and deduplication.
- `data_raw` contains the raw datasets. After preprocessing with process_data.py, the processed datasets are saved in `data_train`.
- `PI1M_unique.csv` is used for model pretraining.
- `EPS_combined_unique.csv` and `Tg_combined_unique.csv` are used for fine-tuning on the $k$ and $T_g$ prediction tasks, respectively.
- `polyimides_pred_prop_base_str_std.csv` is used for polymer structure optimization.

## Usage

Run the pretraining script with the desired model configuration:

### Pretraining
```
python3 polymerCL_Pretrain.py \
    --pretrain_epoch 50 \
    --model_dir models_polymers \
    --n_layers 3 \
    --hidden_dim 128 \
    --dropout 0.2 \
    --lr_pre 1e-4 \
    --batch_size_pre 128 \
    --contrastive 'view1' \
    --seed 221
```
Main arguments:
+ `--pretrain_epoch`: Number of pretraining epochs.
+ `--model_dir_pre`: Directory used to save pretrained model checkpoints.
+ `--n_layers`: Number of graph neural network layers.
+ `--hidden_dim`: Dimension of the hidden representations.
+ `--dropout`: Dropout rate.
+ `--lr_pre`: Learning rate for pretraining.
+ `--batch_size_pre`: Batch size used during pretraining.
+ `--contrastive`: Contrastive-view augmentation strategy. 
  +  `view1` to generate two independently randomized SMILES representations of each molecule. 
  + `view2` applies atom/bond dropping or feature masking.
  + `view12` combines SMILES randomization with one randomly selected graph-level augmentation.
+ `--seed`: Random seed (integer). If omitted, a random integer between 1 and 1000 is selected.

### Fine-Tuning
After pretraining, run the fine-tuning script for the target property:
```
python3 polymerCL_FT_kfold.py \
    --pretrain_epoch 50 \
    --model_dir_pre models_polymers \
    --target EPS \
    --n_layers 3 \
    --hidden_dim 128 \
    --dropout 0.2 \
    --lr 1e-5 \
    --batch_size 4 \
    --n_enum 12 \
    --seed 221
```
Main arguments:
+ `--pretrain_epoch`: Pretraining checkpoint epoch to load.
+ `--model_dir_pre`: Directory containing the pretrained model checkpoints
+ `--target`: Target property for fine-tuning. Supported values are EPS and Tg.
+ `--n_layers`: Number of graph neural network layers.
+ `--hidden_dim`: Dimension of the hidden representations.
+ `--dropout`: Dropout rate.
+ `--lr`: Learning rate for fine-tuning.
+ `--batch_size`: Batch size used during fine-tuning.
+ `--n_enum`: Number of molecular enumerations used for data augmentation.
+ `--seed`: Random seed (integer). If omitted, a random integer between 1 and 1000 is selected.

### Substituent Attribution Analysis

```
python FgAttrAnalysisSDEmultipro.py
```

### Structure Optimization

```
python3 polymerDesignGAs.py --seed 2025
```
- `--seed`: Random seed. If not specified or specified as `None`, a random integer between 1 and 1000 is selected.

## Computational Resources

The experiments were conducted using an NVIDIA A100 GPU.

## Notes

Results may vary slightly across runs due to differences in random initialization, data partitioning, hyperparameter settings, and hardware-dependent numerical behavior. Even when the same random seed is used, minor numerical variations may arise across different hardware and software environments because of nondeterministic operations and floating-point differences.
