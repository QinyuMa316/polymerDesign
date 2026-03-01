import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from umap import UMAP
import pandas as pd
import numpy as np
import random

# plt.rcParams.update({
#     'font.family': 'Arial',
#     'axes.titlesize': 14,  # 主标题
#     'axes.labelsize': 14,  # 横纵轴标题
#     'xtick.labelsize': 11,  # x刻度
#     'ytick.labelsize': 11  # y刻度
# })


# note: def plot_label_distributions
#  deprecated see polyCL-GNN-FT-811-visdata⭐️.py

# def plot_label_distributions(data_loader, target):
#     plt.rcParams.update({
#         'font.family': 'Arial',
#         'axes.titlesize': 14,  # 主标题
#         'axes.labelsize': 14,  # 横纵轴标题
#         'xtick.labelsize': 12,  # x刻度
#         'ytick.labelsize': 12  # y刻度
#     })
#     os.makedirs('plots', exist_ok=True)
#     # Extract labels from each dataset
#     train_labels = [data.y.item() for data in data_loader.train_dataset]
#     val_labels = [data.y.item() for data in data_loader.val_dataset]
#     test_labels = [data.y.item() for data in data_loader.test_dataset]
#
#     # ============== 新增：所有数据合并 ==============
#     all_labels = train_labels + val_labels + test_labels
#
#     # Set up the plot with soft colors
#     plt.figure(figsize=(10, 4))
#
#     # ============== 第一个子图：各数据集分布 ==============
#     plt.subplot(1, 2, 2)  # 1行2列的第1个子图
#     # Soft color palette
#     colors = {
#         'train': (0.4, 0.7, 0.8, 0.7),  # Soft teal
#         'val': (0.8, 0.6, 0.7, 0.7),  # Soft pink
#         'test': (0.6, 0.8, 0.6, 0.7)  # Soft green
#     }
#     # Plot KDE for each set
#     sns.kdeplot(train_labels, label='Train', color=colors['train'], linewidth=2.5, fill=True)
#     sns.kdeplot(val_labels, label='Validation', color=colors['val'], linewidth=2.5, fill=True)
#     sns.kdeplot(test_labels, label='Test', color=colors['test'], linewidth=2.5, fill=True)
#     # Customize the plot
#     plt.title(f'Label Distribution Across Datasets ({target})', pad=10) # , fontweight='bold', fontsize=12,
#     plt.xlabel('Label Value')#, fontweight='bold', fontsize=11)
#     plt.ylabel('Density')#, fontweight='bold', fontsize=11)
#     plt.legend()
#     sns.despine()
#
#     # ============== 第二个子图：整体数据分布 ==============
#     plt.subplot(1, 2, 1)  # 1行2列的第2个子图
#     # 使用不同的颜色表示整体分布
#     overall_color = (0.7, 0.5, 0.8, 0.7)  # Soft purple
#     # 先绘制柱状图（透明度较高）
#     # sns.histplot(all_labels, stat='density', color=overall_color, alpha=0.3, kde=False, binwidth=binwidth)
#     sns.histplot(all_labels, stat='density', color=overall_color, alpha=0.5,
#                  kde=False, bins=18)
#     ax = plt.gca()
#     sns.kdeplot(all_labels, label='All Data', color=overall_color, alpha=0.3, linewidth=2.5, fill=True)
#     # sns.kdeplot(all_labels, label='KDE', color=overall_color, linewidth=2.5, alpha=0.7)
#
#     # 添加统计信息
#     # mean_val = np.mean(all_labels)
#     # median_val = np.median(all_labels)
#     # plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean: {mean_val:.2f}')
#     # plt.axvline(median_val, color='blue', linestyle=':', linewidth=1.5, label=f'Median: {median_val:.2f}')
#
#     plt.title(f'Overall Label Distribution ({target})')#, fontweight='bold', fontsize=11) , pad = 20
#     plt.xlabel('Label Value')#, fontweight='bold', fontsize=11)
#     plt.ylabel('Density')#, fontweight='bold', fontsize=11)
#     plt.legend()
#     sns.despine()
#
#     # Adjust layout
#     plt.tight_layout()
#     # Save the plot
#     plt.savefig(f'plots/label_distributions_{target}.png', dpi=600)
#     plt.close()
#

def plot_pred_vs_true_all(y_true_train, y_pred_train,
                          y_true_val, y_pred_val,
                          y_true_test, y_pred_test,
                          target,
                          folder_name='plots'
                          ):
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 14,  # 主标题
        'axes.labelsize': 14,  # 横纵轴标题
        'xtick.labelsize': 12,  # x刻度
        'ytick.labelsize': 12  # y刻度
    })

    os.makedirs(folder_name, exist_ok=True)
    plt.figure(figsize=(6, 6))

    # Plot training set in blue
    plt.scatter(y_true_train, y_pred_train, color='#6FA8DC', alpha=0.8, label='Train',
                edgecolor='black', linewidths=0.4)
    # Plot validation set in orange
    plt.scatter(y_true_val, y_pred_val, color='#FFA500', alpha=0.8, label='Validation',
                edgecolor='black', linewidths=0.4)
    # Plot test set in green
    plt.scatter(y_true_test, y_pred_test, color='#4CAF50', alpha=0.8, label='Test',
                edgecolor='black', linewidths=0.4)

    # 绘制对角线
    min_val = min(min(y_true_train), min(y_pred_train),
                  min(y_true_val), min(y_pred_val),
                  min(y_true_test), min(y_pred_test))
    max_val = max(max(y_true_train), max(y_pred_train),
                  max(y_true_val), max(y_pred_val),
                  max(y_true_test), max(y_pred_test))
    plt.plot([min_val, max_val], [min_val, max_val], '--', lw=1.5, color='black')

    plt.xlabel('True Values')#, fontweight='bold', fontsize=11)
    plt.ylabel('Predictions')#, fontweight='bold', fontsize=11)
    plt.title(f'Predicted vs True ({target})')#, fontweight='bold', fontsize=11)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f'{folder_name}/pred_vs_true_{target}_all.png', dpi=600)
    plt.close()

def plot_pred_vs_true_all_2(y_true_train, y_pred_train,
                            y_true_val,   y_pred_val,
                            y_true_test,  y_pred_test,
                            target,
                            folder_name = 'plots'
                            ):
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 14,  # 主标题
        'axes.labelsize': 14,  # 横纵轴标题
        'xtick.labelsize': 12,  # x刻度
        'ytick.labelsize': 12  # y刻度
    })
    # fig_size = (6, 8)
    fig_size = (5, 7)
    if target == 'EPS':
        target = 'k'

    os.makedirs(folder_name, exist_ok=True)

    # --- 修改：创建上下两个子图（3:1 高度比例） ---
    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, sharex=True, figsize=fig_size,
        gridspec_kw={'height_ratios': [2, 1]}
    )

    # ========== Panel A：Predicted vs Measured ==========
    # 训练集
    ax_top.scatter(y_true_train, y_pred_train,
                   color='#6FA8DC', alpha=0.8, label='Train',
                   edgecolor='black', linewidths=0.4)
    # 验证集
    ax_top.scatter(y_true_val, y_pred_val,
                   color='#FFA500', alpha=0.8, label='Validation',
                   edgecolor='black', linewidths=0.4)
    # 测试集
    ax_top.scatter(y_true_test, y_pred_test,
                   color='#4CAF50', alpha=0.8, label='Test',
                   edgecolor='black', linewidths=0.4)

    # 对角线及包络线
    min_val = min(min(y_true_train), min(y_pred_train),
                  min(y_true_val),   min(y_pred_val),
                  min(y_true_test),  min(y_pred_test))
    max_val = max(max(y_true_train), max(y_pred_train),
                  max(y_true_val),   max(y_pred_val),
                  max(y_true_test),  max(y_pred_test))

    ax_top.plot([min_val, max_val], [min_val, max_val],
                '--', lw=1.5, color='black')

    # offset = 5                      # --- 修改：包络线偏移（K）；可按需求调整 ---
    # ax_top.plot([min_val, max_val], [min_val + offset, max_val + offset],
    #             '--', lw=1.0, color='black')
    # ax_top.plot([min_val, max_val], [min_val - offset, max_val - offset],
    #             '--', lw=1.0, color='black')

    ax_top.set_ylabel(f'{target} Predictions')
    # ax_top.legend(loc='upper left', frameon=False)
    plt.title(f'Predicted vs True ({target})')
    ax_top.legend(loc='upper left')
    # --- 修改：面板标记 “A” ---
    # ax_top.text(0.97, 0.07, 'A', transform=ax_top.transAxes,
    #             ha='right', va='bottom', fontsize=14, fontweight='bold')

    # ========== Panel B：Residuals ==========
    delta_T_train = np.array(y_pred_train) - np.array(y_true_train)
    delta_T_val   = np.array(y_pred_val)   - np.array(y_true_val)
    delta_T_test  = np.array(y_pred_test)  - np.array(y_true_test)

    ax_bot.scatter(y_true_train, delta_T_train,
                   color='#6FA8DC', alpha=0.8, label='Train',
                   edgecolor='black', linewidths=0.4)
    ax_bot.scatter(y_true_val,   delta_T_val,
                   color='#FFA500', alpha=0.8, label='Validation',
                   edgecolor='black', linewidths=0.4)
    ax_bot.scatter(y_true_test,  delta_T_test,
                   color='#4CAF50', alpha=0.8, label='Test',
                   edgecolor='black', linewidths=0.4)

    ax_bot.axhline(0, ls='--', lw=1.5, color='black')
    ax_bot.set_xlabel(f'{target} True Value')
    ax_bot.set_ylabel(fr'$\Delta {target}$')
    # 紧凑布局 & 保存
    plt.tight_layout(h_pad=0.4)
    fig.savefig(f'{folder_name}/pred_vs_true_{target}_all_v2.png', dpi=600)
    plt.close(fig)


