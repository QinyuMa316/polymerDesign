import pandas as pd
from MolOpt.molDataLoader import MolDataLoader, MolDataset
from torch_geometric.data import Data, Dataset, DataLoader
from MolOpt.GATmodel import GATModel
from MolOpt.CLloss import ContrastiveLoss
import os
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np
from MolOpt.plotData import (
                             # plot_label_distributions,
                             plot_pred_vs_true_all,
                             plot_pred_vs_true_all_2,
                             )
from MolOpt.utils import set_seed #, parse_arguments # log_message
from MolOpt.GNNtrain import test, evaluate, train
import argparse

import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--target', type=float, default=1e-4)
    parser.add_argument('--pretrain_epoch', type=int, required=True)
    parser.add_argument('--model_dir', type=str, required=True)

    return parser.parse_args()

if __name__ == '__main__':

    # args = parse_arguments()
    # target_col = args.target
    # PRE_TREAIN_EPOCH = args.pretrain_epoch
    # model_dir = args.model_dir

    target_col = 'Tg'
    PRE_TREAIN_EPOCH = 50
    model_dir = 'models_polymer'

    # target_col = 'EPS'
    # PRE_TREAIN_EPOCH = 50
    n_enum = 12
    batch_size = 4

    # tg - seed = 56, eps - seed = 88
    if target_col == 'Tg':
        seed = 56 # - 0.74
    elif target_col == 'EPS':
        seed = 78 # 88
    else:
        seed = 42

    set_seed(seed=seed)
    training = True
    patience = 20
    testing = True

    print(f'Seed: {seed}, Trainning: {training}, Testing: {testing}')

    LR = 1e-4
    EPOCH = 1000
    WD = 1e-5
    # batch_size = 4
    # n_enum = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    # model_dir = 'models_polymer'
    # os.makedirs(model_dir, exist_ok=True)
    # ----------------------------------------------------------------------------------------
    n_layers = 3
    hidden_dim = 128
    num_heads = 4
    dropout = 0.2
    print(f'Model Architecture: Num Layer: {n_layers}, Hidden Dim: {hidden_dim}, Num Heads: {num_heads}')
    model = GATModel(hidden_dim=hidden_dim, dropout=dropout, num_layers=n_layers)
    # ----------------------------------------------------------------------------------------

    smiles_col = 'SMILES'
    data_path = f'data/{target_col}_combined.csv'
    print(f'Target: {target_col}')
    data = pd.read_csv(data_path)
    smiles_list = data[smiles_col].tolist()
    label_list = data[target_col].tolist()
    print(f'Total data size for training: {len(data)}')
    data_loader = MolDataLoader(smiles_list, label_list,
                                batch_size=batch_size, shuffle=True)
    train_loader, val_loader, test_loader = data_loader.get_split_loaders(test_size=0.1, val_size=0.1,
                                                                          n_enum=n_enum,
                                                                          random_state = seed
                                                                          )
    # plot_label_distributions(data_loader, target_col)
    # ----------------------------------------------------------------------------------------

    model_path = f'{model_dir}/best_model_{target_col}_cl_{PRE_TREAIN_EPOCH}epo_aug{n_enum}.pth'
    print(f'fine-tuning Hyperparams: LR: {LR}, WD: {WD}, batch size: {batch_size}, data augmentation: {n_enum}, dropout rate: {dropout}')

    # ----------------------------------------------------------------------------------------
    if not os.path.exists(model_path):
        print(f'Fail to find model at {model_path}, start to load pretained model')
        # note:
        pre_model_path = f'{model_dir}/pretrain_encoder_150_{PRE_TREAIN_EPOCH}epoch.pth'
        if os.path.exists(pre_model_path):
            checkpoint = torch.load(pre_model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Pretrained Model successfully loaded from {pre_model_path}')
        else:
            print(f'Pretrained Model Not Found from {pre_model_path}')
    model.to(device)

    # ----------------------------------------------------------------------------------------

    if training:
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
        criterion = nn.MSELoss()
        print(f"Starting training ...")
        model = train(model, train_loader, val_loader, optimizer, criterion, device,
                      epochs=EPOCH, patience=patience, resume=True,
                      model_path=model_path,
                      contrastive=False
                      )

    if testing:
        print("Evaluating on test set...")
        y_true_train, y_pred_train = test(model, train_loader, device)
        y_true_val, y_pred_val = test(model, val_loader, device)
        y_true_test, y_pred_test = test(model, test_loader, device)

        # y_true_test, y_pred_test = test(model, test_loader, device)
        r2 = r2_score(y_true_test, y_pred_test)
        mae = mean_absolute_error(y_true_test, y_pred_test)
        mse = mean_squared_error(y_true_test, y_pred_test)
        rmse = np.sqrt(mse)
        print(f'Test R2: {r2:.4f}, Test MAE: {mae:.4f}, Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}')
        # plot_pred_vs_true(y_true_test, y_pred_test, target_col)
        plot_pred_vs_true_all(y_true_train, y_pred_train,
                              y_true_val, y_pred_val,
                              y_true_test, y_pred_test,
                              target_col)
        plot_pred_vs_true_all_2(y_true_train, y_pred_train,
                              y_true_val, y_pred_val,
                              y_true_test, y_pred_test,
                              target_col)

