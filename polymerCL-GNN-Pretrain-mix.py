import pandas as pd
from MolOpt.molDataLoader import MolDataLoader
from MolOpt.GATmodel import GATModel
from MolOpt.CLloss import ContrastiveLoss
import os
import torch
from MolOpt.utils import set_seed
from MolOpt.GNNtrain import train
import argparse
import warnings
warnings.filterwarnings("ignore")

def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr_pre', type=float, default=1e-4)
    parser.add_argument('--batch_size_pre', type=int, default=256)
    parser.add_argument('--model_dir', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':

    args = parse_arguments()
    lr_pre = args.lr_pre
    batch_size_pre = args.batch_size_pre
    model_dir = args.model_dir

    seed = 42
    set_seed(seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    os.makedirs(model_dir, exist_ok=True)
    # ----------------------------------------------------------------------------------------
    n_layers = 3
    hidden_dim = 128
    num_heads = 4
    dropout = 0.2
    print(f'Model Architecture: Num Layer: {n_layers}, Hidden Dim: {hidden_dim}, Num Heads: {num_heads}')
    model = GATModel(hidden_dim=hidden_dim, dropout=dropout, num_layers=n_layers)
    model.to(device)
    # ----------------------------------------------------------------------------------------
    PRETRAIN = True
    PRE_EPOCH = 100
    wd_pre = 1e-5
    # lr_pre = 1e-4
    # batch_size_pre = 256
    # note: 理论上对比学习batch_size越大越好
    print(f'Contrastive Learning Hyperparameters: LR: {lr_pre}, WD: {wd_pre}, Batch Size: {batch_size_pre}')
    # ----------------------------------------------------------------------------------------

    # ========== 对比学习数据集 ==========
    data_path_pre = 'data/PI1M.csv'
    smiles_col = 'SMILES'
    data_pre = pd.read_csv(data_path_pre)
    smiles_list_pre = data_pre[smiles_col].tolist()
    print(f'Total data_mod size for pretraining: {len(data_pre)}')
    # 创建对比学习数据加载器
    pre_loader = MolDataLoader(smiles_list_pre, [0] * len(smiles_list_pre),  # 伪标签
                               batch_size=batch_size_pre, shuffle=True)
    # 使用数据增强生成多个视图
    train_loader_pre, _, _ = pre_loader.get_split_loaders_cl(test_size=0, val_size=0, )
    # ----------------------------------------------------------------------------------------

    pre_model_path = f'{model_dir}/pretrain_encoder.pth'
    # if os.path.exists(pre_model_path):
    #     checkpoint = torch.load(pre_model_path, map_location=device)
    #     model.load_state_dict(checkpoint['model_state_dict'])
    #     print(f'Pretrained Model successfully loaded from {pre_model_path}')
    # else:
    #     print(f'Pretrained Model Not Found from {pre_model_path}')
    # # ----------------------------------------------------------------------------------------

    if PRETRAIN:
        optimizer_pre = torch.optim.Adam(model.parameters(), lr=lr_pre, weight_decay=wd_pre)
        criterion_pre = ContrastiveLoss(temperature=0.1)

        train(model, train_loader_pre, None,
              optimizer_pre, criterion_pre, device,
              epochs=PRE_EPOCH, patience=PRE_EPOCH + 1,  # 不早停
              model_path=pre_model_path, contrastive=True)
        # 保存预训练模型
        # torch.save(model.state_dict(), pre_model_path)
        print(f"Pretraining completed.")

