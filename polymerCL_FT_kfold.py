import pandas as pd
import random
from MolOpt.molDataLoader import MolDataLoader, MolDataset
from torch_geometric.data import Data, Dataset, DataLoader
from MolOpt.GATmodel import GATModel
from MolOpt.CLloss import ContrastiveLoss
import os
import time
import torch
import torch.nn as nn
from contextlib import redirect_stdout, redirect_stderr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import KFold # === k fold ===
import numpy as np
from MolOpt.plotData import plot_pred_vs_true_all, plot_pred_vs_true_all_2
from MolOpt.utils import set_seed #, parse_arguments # log_message
from MolOpt.GNNtrain import test, evaluate, train
import argparse
from tqdm import tqdm
import logging
import warnings
warnings.filterwarnings("ignore")


def format_elapsed_time(elapsed_seconds):
    elapsed_seconds = int(elapsed_seconds)
    hours, remainder = divmod(elapsed_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

def parse_arguments():

    parser = argparse.ArgumentParser()
    parser.add_argument('--pretrain_epoch', type=int, required=True)
    parser.add_argument('--model_dir_pre', type=str, required=True)
    parser.add_argument('--target', type=str, required=True)

    parser.add_argument('--n_layers', type=int, default=3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.02)
    parser.add_argument('--lr', type=float, default=1e-5)

    parser.add_argument('--epoch_pre', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--n_enum', type=int, default=12)

    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()

    PRE_TRAIN_EPOCH = args.pretrain_epoch
    model_dir_pre = args.model_dir_pre
    target_col = args.target
    # PRE_TRAIN_EPOCH = 50
    model_dir_pre = f'models_polymer'
    # model_dir = 'models_polymers'
    # target = 'Tg'

    n_layers = args.n_layers
    hidden_dim = args.hidden_dim
    dropout = args.dropout
    lr = args.lr
    batch_size = args.batch_size
    n_enum = args.n_enum

    program_start_time = time.time()

    # n_layers = 3
    # hidden_dim = 128
    num_heads = 4
    # dropout = 0.2
    # n_enum = 12
    # batch_size = 4

    seed = random.randint(1, 1000)
    # seed = 221
    # target_col = 'EPS'
    # target_col = 'Tg'
    # Seed: 221, Training: True, Testing: True


    # tg - seed = 56, eps - seed = 88
    # if target_col == 'Tg':
    #     seed = 56 # - 0.74
    # elif target_col == 'EPS':
    #     seed = 78 # 88
    # else:
    #     seed = 42

    # seed = 812
    set_seed(seed=seed)
    training = True
    patience = 20
    testing = True

    model_dir = f'{model_dir_pre}_10fold_{target_col}_seed{seed}'
    os.makedirs(model_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(model_dir, f"10fold_{target_col}_seed{seed}.txt"),
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    train_log_path = os.path.join(model_dir, f"10fold_{target_col}_seed{seed}_train.txt")

    logging.info(f'Seed: {seed}, Training: {training}, Testing: {testing}')

    LR = 1e-5
    EPOCH = 1000
    WD = 1e-5
    # batch_size = 4
    # n_enum = 8

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')
    # model_dir = 'models_polymer'
    # os.makedirs(model_dir, exist_ok=True)
    # ----------------------------------------------------------------------------------------

    logging.info(f'Model Architecture: Num Layer: {n_layers}, Hidden Dim: {hidden_dim}, Num Heads: {num_heads}')
    # === k fold ===
    # ----------------------------------------------------------------------------------------

    smiles_col = 'SMILES'
    data_path = f'PolyDesign_data/data_train/{target_col}_combined_unique.csv'
    logging.info(f'Target: {target_col}')
    data = pd.read_csv(data_path)
    smiles_list = data[smiles_col].tolist()
    label_list = data[target_col].tolist()
    logging.info(f'Total data size for training: {len(data)}')
    data_loader = MolDataLoader(smiles_list, label_list, batch_size=batch_size, shuffle=True)
    unique_smiles = np.array(list(dict.fromkeys(smiles_list)))
    kf = KFold(n_splits=10, shuffle=True, random_state=seed)
    def make_loader(idx, shuffle=False):
        s, l = [smiles_list[i] for i in idx], [label_list[i] for i in idx]
        if n_enum > 0: s, l = data_loader._augment(s, l, n_enum)
        return DataLoader(MolDataset(s, l, contrastive=False), batch_size=batch_size, shuffle=shuffle)
    fold_metrics = []
    # === k fold ===
    # plot_label_distributions(data_loader, target_col)
    # ----------------------------------------------------------------------------------------

    model_path = f'{model_dir}/best_model_{target_col}_cl_{PRE_TRAIN_EPOCH}epo_aug{n_enum}.pth'
    logging.info(f'fine-tuning Hyperparams: LR: {LR}, WD: {WD}, batch size: {batch_size}, data augmentation: {n_enum}, dropout rate: {dropout}')

    # ----------------------------------------------------------------------------------------
    # === k fold ===
    fold_iterator = tqdm(
        enumerate(kf.split(unique_smiles), 1),
        total=10,
        desc='10-Fold CV',
        unit='fold'
    )
    for fold, (train_val_gidx, test_gidx) in fold_iterator:
        fold_start_time = time.time()
        fold_iterator.set_postfix_str(f'Fold {fold}/10')
        logging.info(f'========== Fold {fold}/10 ==========')
        train_val_groups, test_groups = set(unique_smiles[train_val_gidx]), set(unique_smiles[test_gidx])
        train_val_idx = np.array([i for i, s in enumerate(smiles_list) if s in train_val_groups])
        inner_groups = np.array(list(dict.fromkeys([smiles_list[i] for i in train_val_idx])))
        train_gidx, val_gidx = next(KFold(n_splits=9, shuffle=True, random_state=seed + fold).split(inner_groups))
        train_groups, val_groups = set(inner_groups[train_gidx]), set(inner_groups[val_gidx])
        train_idx = np.array([i for i in train_val_idx if smiles_list[i] in train_groups])
        val_idx = np.array([i for i in train_val_idx if smiles_list[i] in val_groups])
        test_idx = np.array([i for i, s in enumerate(smiles_list) if s in test_groups])
        train_loader, val_loader, test_loader = make_loader(train_idx, True), make_loader(val_idx), make_loader(test_idx)
        fold_model_path = model_path.replace('.pth', f'_fold{fold}.pth')
        model = GATModel(hidden_dim=hidden_dim, dropout=dropout, num_layers=n_layers)
        if not os.path.exists(fold_model_path):
            logging.info(f'Fail to find model at {fold_model_path}, start to load pretrained model')
            # note:
            pre_model_path = f'{model_dir_pre}/pretrain_encoder_{PRE_TRAIN_EPOCH}epoch.pth'
            if os.path.exists(pre_model_path):
                checkpoint = torch.load(pre_model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                logging.info(f'Pretrained Model successfully loaded from {pre_model_path}')
            else:
                logging.info(f'Pretrained Model Not Found from {pre_model_path}')
        model.to(device)
        # ----------------------------------------------------------------------------------------

        if training:
            optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
            criterion = nn.MSELoss()
            logging.info(f"Starting training ...")
            with open(train_log_path, 'a', buffering=1) as train_log_file, redirect_stdout(train_log_file), redirect_stderr(train_log_file):
                print(f'========== Fold {fold}/10 ==========')
                model = train(model, train_loader, val_loader, optimizer, criterion, device,
                              epochs=EPOCH, patience=patience, resume=True,
                              model_path=fold_model_path,
                              contrastive=False
                              )

        if testing:
            logging.info("Evaluating on test set...")
            y_true_train, y_pred_train = test(model, train_loader, device)
            y_true_val, y_pred_val = test(model, val_loader, device)
            y_true_test, y_pred_test = test(model, test_loader, device)

            # y_true_test, y_pred_test = test(model, test_loader, device)
            r2 = r2_score(y_true_test, y_pred_test)
            mae = mean_absolute_error(y_true_test, y_pred_test)
            mse = mean_squared_error(y_true_test, y_pred_test)
            rmse = np.sqrt(mse)
            fold_metrics.append([r2, mae, mse, rmse])
            logging.info(f'Fold {fold} Test R2: {r2:.4f}, Test MAE: {mae:.4f}, Test MSE: {mse:.4f}, Test RMSE: {rmse:.4f}')
            # plot_pred_vs_true(y_true_test, y_pred_test, target_col)
            plot_pred_vs_true_all(y_true_train, y_pred_train,
                                  y_true_val, y_pred_val,
                                  y_true_test, y_pred_test,
                                  f'{target_col}_fold{fold}')
            plot_pred_vs_true_all_2(y_true_train, y_pred_train,
                                  y_true_val, y_pred_val,
                                  y_true_test, y_pred_test,
                                  f'{target_col}_fold{fold}')
        fold_elapsed_time = time.time() - fold_start_time
        fold_time_message = f'Fold {fold}/10 finished in {format_elapsed_time(fold_elapsed_time)} ({fold_elapsed_time:.2f}s)'
        print(fold_time_message, flush=True)
        logging.info(fold_time_message)
    if fold_metrics:
        fold_metrics = np.array(fold_metrics)
        logging.info(f'10-Fold CV Mean R2: {fold_metrics[:, 0].mean():.4f} +/- {fold_metrics[:, 0].std():.4f}, MAE: {fold_metrics[:, 1].mean():.4f} +/- {fold_metrics[:, 1].std():.4f}, MSE: {fold_metrics[:, 2].mean():.4f} +/- {fold_metrics[:, 2].std():.4f}, RMSE: {fold_metrics[:, 3].mean():.4f} +/- {fold_metrics[:, 3].std():.4f}')
    total_elapsed_time = time.time() - program_start_time
    total_time_message = f'All program finished in {format_elapsed_time(total_elapsed_time)} ({total_elapsed_time:.2f}s)'
    print(total_time_message, flush=True)
    logging.info(total_time_message)
    # === k fold ===
