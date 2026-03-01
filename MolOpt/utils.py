
import json

import pandas as pd
import random
import torch
from torch_geometric import seed_everything
import numpy as np
import argparse

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    seed_everything(seed)

def log_message(message, file='out.txt'):
    """将消息写入文件（不打印到控制台）"""
    with open(file, 'a') as f:
        f.write(message + '\n')

def saveResults(all_results, file_path):
    # 保存指标结果
    metrics_df = pd.DataFrame([{
        'fold': r['fold'],
        'R2': r['r2'],
        'MAE': r['mae'],
        'MSE': r['mse'],
        'RMSE': r['rmse']
    } for r in all_results])
    # metrics_df.to_csv(f'{model_dir}/{target_col}_cv_metrics.csv', index=False)

    # 计算并保存平均指标
    avg_r2 = np.mean([r['r2'] for r in all_results])
    avg_mae = np.mean([r['mae'] for r in all_results])
    avg_mse = np.mean([r['mse'] for r in all_results])
    avg_rmse = np.mean([r['rmse'] for r in all_results])

    avg_row = pd.DataFrame([{
        'fold': 'average',
        'R2': avg_r2,
        'MAE': avg_mae,
        'MSE': avg_mse,
        'RMSE': avg_rmse
    }])
    # avg_metrics.to_csv(f'{model_dir}/{target_col}_cv_avg_metrics.csv', index=False)
    # 合并原始指标和平均指标
    combined_df = pd.concat([metrics_df, avg_row], ignore_index=True)
    # 保存到单个CSV文件
    combined_df.to_csv(file_path, index=False)

    print(f'\n{"=" * 40}')
    print(f'Cross Validation Average Results:')
    print(f'Average R2: {avg_r2:.4f}, Average MAE: {avg_mae:.4f}, Average MSE: {avg_mse:.4f}, Average RMSE: {avg_rmse:.4f}')


def parse_arguments():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Process PDFs and extract reactions.")
    parser.add_argument('--target', type=str, required=True)
    parser.add_argument('--aug', type=int, required=True)
    parser.add_argument('--pretrain_epoch', type=int, required=False)
    parser.add_argument('--batch_size', type=int, required=True)
    return parser.parse_args()




