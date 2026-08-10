import pandas as pd
from MolOpt.molDataLoader import MolDataLoader
from MolOpt.GATmodel import GATModel
import os
import torch
import json
import numpy as np
from MolOpt.utils import set_seed
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from umap import UMAP
from matplotlib import colors
import warnings
from tqdm import tqdm
from rdkit import Chem

warnings.filterwarnings("ignore")


# ==== cache ====
# Cache settings
CACHE_DIR = 'umap/cache'
FEATURE_CACHE_DIR = os.path.join(CACHE_DIR, 'features')
UMAP_CACHE_DIR = os.path.join(CACHE_DIR, 'embeddings')
CACHE_VERSION = 'v1'

# 平时修改颜色、透明度、legend、title等绘图代码时全部保持 False
FORCE_REBUILD_BG_FEATURES = False
FORCE_REBUILD_SMALL_FEATURES = False
FORCE_REBUILD_UMAP = False

# UMAP参数
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_N_EPOCHS = 200
UMAP_RANDOM_STATE = 42

os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(FEATURE_CACHE_DIR, exist_ok=True)
os.makedirs(UMAP_CACHE_DIR, exist_ok=True)


def get_bg_feature_cache_path(target):
    return os.path.join(FEATURE_CACHE_DIR, f'{target}_background_features_{CACHE_VERSION}.npz')


def get_small_feature_cache_path(target, name):
    return os.path.join(FEATURE_CACHE_DIR, f'{target}_{name}_features_{CACHE_VERSION}.npz')


def get_umap_cache_path(target):
    min_dist_tag = str(UMAP_MIN_DIST).replace('.', 'p')
    return os.path.join(UMAP_CACHE_DIR, f'umap_{target}_nn{UMAP_N_NEIGHBORS}_md{min_dist_tag}_ep{UMAP_N_EPOCHS}_{CACHE_VERSION}.npz')


def background_feature_cache_exists(target):
    return os.path.exists(get_bg_feature_cache_path(target)) or os.path.exists(f'{target}_feats.json')


def complete_umap_cache_exists(target):
    cache_path = get_umap_cache_path(target)
    if not os.path.exists(cache_path):
        return False

    required_keys = ['embedding', 'labels_plot', 'embedding_pi', 'embedding_pi_mod', 'embedding_prop']

    try:
        with np.load(cache_path) as cache:
            return all(key in cache.files for key in required_keys)
    except Exception:
        return False
# ==== cache ====


def get_features(model, loader, device, desc='Computing features', cache_path=None, force_rebuild=False):
    """
    获取小数据集的GNN特征。
    如果cache_path存在，则优先直接读取缓存。
    """

    # ==== cache ====
    if cache_path is not None and os.path.exists(cache_path) and not force_rebuild:
        print(f'Loading cached features from {cache_path}')
        with np.load(cache_path) as data:
            features = data['features']
            labels = data['labels']
        return features, labels
    # ==== cache ====

    if loader is None:
        raise ValueError('loader is required when feature cache does not exist.')

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, unit='batch'):
            batch = batch.to(device)
            out, readout = model(batch, return_cl_dim=False)
            features.append(readout.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # ==== cache ====
    if cache_path is not None:
        np.savez(cache_path, features=features, labels=labels)
        print(f'Features cached to {cache_path}')
    # ==== cache ====

    return features, labels


def get_features2(model, loader, device, target_col):
    """
    获取988775条背景数据的GNN特征。
    优先读取NPZ缓存。
    如果只有以前的Tg_feats.json/EPS_feats.json，则自动读取并转成NPZ。
    """

    # ==== cache ====
    feat_path = get_bg_feature_cache_path(target_col)
    legacy_json_path = f'{target_col}_feats.json'

    if os.path.exists(feat_path) and not FORCE_REBUILD_BG_FEATURES:
        print(f'Loading cached background features from {feat_path}')
        with np.load(feat_path) as data:
            features = data['features']
            labels = data['labels']
        return features, labels

    # 兼容以前已经生成的 Tg_feats.json / EPS_feats.json
    if os.path.exists(legacy_json_path) and not FORCE_REBUILD_BG_FEATURES:
        print(f'Loading legacy feature cache from {legacy_json_path}')
        with open(legacy_json_path, 'r') as f:
            data = json.load(f)

        features = np.array(data['features'])
        labels = np.array(data['labels'])

        print(f'Converting legacy JSON cache to faster NPZ cache: {feat_path}')
        np.savez(feat_path, features=features, labels=labels)
        return features, labels
    # ==== cache ====

    if loader is None:
        raise ValueError('loader is required because background feature cache does not exist.')

    print(f'{feat_path} does not exist, start to compute background features ...')

    model.eval()
    features = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f'Computing {target_col} background features', unit='batch'):
            batch = batch.to(device)
            out, readout = model(batch, return_cl_dim=False)
            features.append(readout.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    features = np.concatenate(features)
    labels = np.concatenate(labels)

    # ==== cache ====
    np.savez(feat_path, features=features, labels=labels)
    print(f'Background features cached to {feat_path}')
    # ==== cache ====

    return features, labels


def visualize_with_umap3(features, labels, target,
                         features_pi=None, labels_pi=None,
                         features_pi_mod=None, labels_pi_mod=None,
                         features_prop=None, labels_prop=None):

    scat_size = 30
    plt.rcParams.update({'font.size': 16})
    np.random.seed(42)
    os.makedirs('umap', exist_ok=True)

    # ==== cache ====
    umap_cache_path = get_umap_cache_path(target)

    if complete_umap_cache_exists(target) and not FORCE_REBUILD_UMAP:
        print('=' * 60)
        print(f'Loading cached UMAP embeddings from {umap_cache_path}')
        print('=' * 60)

        with np.load(umap_cache_path) as cache:
            embedding = cache['embedding']
            labels_plot = cache['labels_plot']
            embedding_pi = cache['embedding_pi']
            embedding_pi_mod = cache['embedding_pi_mod']
            embedding_prop = cache['embedding_prop']

        print(f'Background UMAP: {embedding.shape}')
        print(f'Initial PI UMAP: {embedding_pi.shape}')
        print(f'Optimized PI UMAP: {embedding_pi_mod.shape}')
        print(f'Trained labeled UMAP: {embedding_prop.shape}')

    else:
        if features is None or labels is None:
            raise ValueError('features and labels are required because UMAP cache does not exist.')

        print('=' * 60)
        print('UMAP cache not found, start computing UMAP ...')
        print('=' * 60)

        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        # 对输入数据排序，确保输入顺序稳定
        sort_idx = np.argsort(features_scaled.sum(axis=1))
        features_scaled = features_scaled[sort_idx]
        labels_plot = labels[sort_idx]

        reducer = UMAP(random_state=UMAP_RANDOM_STATE, n_neighbors=UMAP_N_NEIGHBORS, min_dist=UMAP_MIN_DIST,
                       metric='euclidean', n_jobs=-1, transform_seed=42, n_epochs=UMAP_N_EPOCHS, verbose=True)

        embedding = reducer.fit_transform(features_scaled)

        embedding_pi = np.empty((0, 2))
        embedding_pi_mod = np.empty((0, 2))
        embedding_prop = np.empty((0, 2))

        if features_pi is not None and labels_pi is not None:
            print('Transforming Initial PI Structures ...')
            embedding_pi = reducer.transform(scaler.transform(features_pi))

        if features_pi_mod is not None and labels_pi_mod is not None:
            print('Transforming Optimized PI Structures ...')
            embedding_pi_mod = reducer.transform(scaler.transform(features_pi_mod))

        if features_prop is not None and labels_prop is not None:
            print('Transforming Trained Labeled Structures ...')
            embedding_prop = reducer.transform(scaler.transform(features_prop))

        # ==== cache ====
        np.savez(umap_cache_path,
                 embedding=embedding,
                 labels_plot=labels_plot,
                 embedding_pi=embedding_pi,
                 embedding_pi_mod=embedding_pi_mod,
                 embedding_prop=embedding_prop)

        print('=' * 60)
        print(f'UMAP embeddings cached to {umap_cache_path}')
        print('=' * 60)
        # ==== cache ====

    # ============================================================
    # 从这里开始全部只是绘图
    # 修改下面的颜色、透明度、legend、title等不会重新计算UMAP
    # ============================================================

    plt.figure(figsize=(10, 8))

    norm = colors.LogNorm(vmin=labels_plot.min(), vmax=labels_plot.max())

    # Pre-training Polymer Data
    scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                          c=labels_plot, cmap='viridis', norm=norm,
                          alpha=0.2, s=scat_size, linewidths=0)

    cbar = plt.colorbar(scatter)

    cbar.set_label(r'Pre-training Structures (log scale)', labelpad=10)
    # if target == 'Tg':
    #     cbar.set_label(r'Pre-training Structures ($T_g$, log scale)', labelpad=10)
    # elif target == 'EPS':
    #     cbar.set_label(r'Pre-training Structures ($k$, log scale)', labelpad=10)
    # else:
    #     cbar.set_label(f'Pre-training Structures ({target}, log scale)', labelpad=10)

    if target == 'EPS':
        plt.title('UMAP Visualization of Features (k)')
    elif target == 'Tg':
        plt.title(r'UMAP Visualization of Features ($T_g$)')
    else:
        plt.title(f'UMAP Visualization of Features ({target})')

    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')

    # 保存只有背景数据的版本
    plt.savefig(f'umap/umap_visualization_{target}_v3_wo_scat.png', dpi=600, bbox_inches='tight')

    # Initial PI Structures
    if embedding_pi.shape[0] > 0:
        plt.scatter(embedding_pi[:, 0], embedding_pi[:, 1],
                    edgecolors='red', facecolors='none', alpha=0.7,
                    s=scat_size, linewidth=1.5, label='Initial PI Structures')

    # Optimized PI Structures
    if embedding_pi_mod.shape[0] > 0:
        plt.scatter(embedding_pi_mod[:, 0], embedding_pi_mod[:, 1],
                    edgecolors='deeppink', facecolors='none', alpha=0.7,
                    s=scat_size, linewidth=1.5, label='Optimized PI Structures')

    # Trained Labeled Structures
    if embedding_prop.shape[0] > 0:
        plt.scatter(embedding_prop[:, 0], embedding_prop[:, 1],
                    edgecolors='blue', facecolors='none', alpha=0.7,
                    s=scat_size, linewidth=1.5, label='Fine-tuning Structures')

    if embedding_pi.shape[0] > 0 or embedding_pi_mod.shape[0] > 0 or embedding_prop.shape[0] > 0:
        plt.legend(loc='upper left', fontsize=14, frameon=True)

    plt.savefig(f'umap/umap_visualization_{target}_v3.png', dpi=600, bbox_inches='tight')
    plt.close()


# ==== cache ====
def load_or_create_valid_data(valid_data_path):
    """
    缓存RDKit过滤后的有效SMILES。
    第一次检查约100万条SMILES，以后直接读取过滤后的CSV。
    """

    if os.path.exists(valid_data_path):
        print(f'Loading cached valid SMILES data from {valid_data_path}')
        return pd.read_csv(valid_data_path)

    data_path_pre = 'umap/PI1M_pred_label_tg_k.csv'
    data_pre = pd.read_csv(data_path_pre)
    valid_indices = []

    for i, smi in enumerate(tqdm(data_pre['SMILES'], desc='Validating SMILES', unit='mol')):
        mol = Chem.MolFromSmiles(str(smi))

        if mol is not None:
            valid_indices.append(i)
        else:
            tqdm.write(f'Skipping invalid SMILES at row {i}: {smi}')

    data_valid = data_pre.iloc[valid_indices].reset_index(drop=True)
    data_valid.to_csv(valid_data_path, index=False)

    print(f'Original molecules: {len(data_pre)}')
    print(f'Valid molecules:    {len(data_valid)}')
    print(f'Removed molecules:  {len(data_pre) - len(data_valid)}')
    print(f'Valid data cached to {valid_data_path}')

    return data_valid
# ==== cache ====


if __name__ == '__main__':

    seed = 42
    set_seed(seed=seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model_dir = 'models_polymer'
    os.makedirs(model_dir, exist_ok=True)

    n_layers = 3
    hidden_dim = 128
    num_heads = 4
    dropout = 0.2

    print(f'Model Architecture: Num Layer: {n_layers}, Hidden Dim: {hidden_dim}, Num Heads: {num_heads}')

    target_cols = ['Tg', 'EPS']
    smiles_col = 'SMILES'
    valid_data_path = 'umap/PI1M_pred_label_tg_k_valid.csv'

    # ==== cache ====
    # 延迟读取100万条valid CSV。
    # 如果Tg/EPS的完整UMAP cache已经存在，连CSV和GNN模型都不用加载。
    data_valid = None
    # ==== cache ====

    for target_col in target_cols:

        print('\n' + '=' * 80)
        print(f'Processing {target_col} ...')
        print('=' * 80)

        # ==== cache ====
        # 最快路径：
        # 如果最终UMAP二维坐标已经缓存，直接画图。
        # 不加载模型、不加载100万features、不运行GNN、不运行UMAP。
        if complete_umap_cache_exists(target_col) and not FORCE_REBUILD_UMAP:
            print(f'Complete UMAP cache found for {target_col}. Skip model/features/UMAP computation.')
            visualize_with_umap3(None, None, target_col)
            print(f'Finish plotting {target_col} directly from UMAP cache.')
            continue
        # ==== cache ====

        # ---------------------------------------------------------
        # UMAP cache不存在，因此才需要模型
        # ---------------------------------------------------------
        model = GATModel(hidden_dim=hidden_dim, dropout=dropout, num_layers=n_layers)

        model_path = f'{model_dir}/best_model_{target_col}_cl_50epo_aug12.pth'

        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f'Model successfully loaded from {model_path}')
        else:
            raise FileNotFoundError(f'Model not found at {model_path}')

        model.to(device)

        # =========================================================
        # Background features: 988775 samples
        # =========================================================

        # ==== cache ====
        # 如果背景feature cache已经存在，就完全不需要创建100万数据的DataLoader。
        if background_feature_cache_exists(target_col) and not FORCE_REBUILD_BG_FEATURES:
            features0, labels0 = get_features2(model, None, device, target_col)

        else:
            if data_valid is None:
                data_valid = load_or_create_valid_data(valid_data_path)

            smiles_list0 = data_valid[smiles_col].tolist()
            label_list0 = data_valid[target_col].tolist()

            dataloader0 = MolDataLoader(smiles_list0, label_list0, batch_size=1, shuffle=True)
            loader0 = dataloader0.get_all_loader()

            features0, labels0 = get_features2(model, loader0, device, target_col)
        # ==== cache ====

        # =========================================================
        # Initial PI Structures
        # =========================================================

        data_path_pi = 'umap/polyimides_pred_prop_tg_k.csv'
        data_pi = pd.read_csv(data_path_pi)

        smiles_list_pi = data_pi[smiles_col].tolist()
        label_list_pi = data_pi[target_col].tolist()

        # ==== cache ====
        pi_cache_path = get_small_feature_cache_path(target_col, 'initial_pi')

        if os.path.exists(pi_cache_path) and not FORCE_REBUILD_SMALL_FEATURES:
            features_pi, labels_pi = get_features(model, None, device, desc='Initial PI', cache_path=pi_cache_path)
        else:
            loader_pi = MolDataLoader(smiles_list_pi, label_list_pi, batch_size=1).get_all_loader()
            features_pi, labels_pi = get_features(model, loader_pi, device, desc='Initial PI', cache_path=pi_cache_path, force_rebuild=FORCE_REBUILD_SMALL_FEATURES)
        # ==== cache ====

        # =========================================================
        # Trained Labeled Structures
        # =========================================================

        data_path_prop = f'umap/{target_col}_wopi.csv'
        data_prop = pd.read_csv(data_path_prop)

        smiles_list_prop = data_prop[smiles_col].tolist()
        label_list_prop = data_prop[target_col].tolist()

        # ==== cache ====
        prop_cache_path = get_small_feature_cache_path(target_col, 'trained_labeled')

        if os.path.exists(prop_cache_path) and not FORCE_REBUILD_SMALL_FEATURES:
            features_prop, labels_prop = get_features(model, None, device, desc='Trained labeled structures', cache_path=prop_cache_path)
        else:
            loader_prop = MolDataLoader(smiles_list_prop, label_list_prop, batch_size=1).get_all_loader()
            features_prop, labels_prop = get_features(model, loader_prop, device, desc='Trained labeled structures', cache_path=prop_cache_path, force_rebuild=FORCE_REBUILD_SMALL_FEATURES)
        # ==== cache ====

        # =========================================================
        # Optimized PI Structures
        # =========================================================

        data_path_pi_mod = 'umap/population_nsga2_seed2025/pareto_frontier_all.csv'
        data_pi_mod = pd.read_csv(data_path_pi_mod)

        smiles_list_pi_mod = data_pi_mod[smiles_col].tolist()
        label_list_pi_mod = data_pi_mod[target_col].tolist()

        # ==== cache ====
        pi_mod_cache_path = get_small_feature_cache_path(target_col, 'optimized_pi')

        if os.path.exists(pi_mod_cache_path) and not FORCE_REBUILD_SMALL_FEATURES:
            features_pi_mod, labels_pi_mod = get_features(model, None, device, desc='Optimized PI', cache_path=pi_mod_cache_path)
        else:
            loader_pi_mod = MolDataLoader(smiles_list_pi_mod, label_list_pi_mod, batch_size=1).get_all_loader()
            features_pi_mod, labels_pi_mod = get_features(model, loader_pi_mod, device, desc='Optimized PI', cache_path=pi_mod_cache_path, force_rebuild=FORCE_REBUILD_SMALL_FEATURES)
        # ==== cache ====

        print('Finish loading features, start UMAP visualization ...')

        visualize_with_umap3(features0, labels0, target_col,
                             features_pi=features_pi, labels_pi=labels_pi,
                             features_pi_mod=features_pi_mod, labels_pi_mod=labels_pi_mod,
                             features_prop=features_prop, labels_prop=labels_prop)

        print(f'Finish plotting UMAP visualization for {target_col}.')

    """
    你真正需要长期保存的，其实是最终的二维 UMAP：
    embedding.shape # (988775, 2)
    
    必须长期保存：
    PI1M_pred_label_tg_k_valid.csv
    → 合法 SMILES，不用重新检查100万次

    umap_Tg_embedding.npz
    umap_EPS_embedding.npz
    → 100万背景点的最终二维坐标
    
    scaler_Tg.joblib
    scaler_EPS.joblib
    → 给未来新数据标准化

    umap_reducer_Tg.joblib
    umap_reducer_EPS.joblib
    → 把未来新数据投影到同一个UMAP空间
    
    可以删除：
    Tg_feats.json           ❌ 可以删除
    EPS_feats.json          ❌ 可以删除
    
    Tg_background_features.npz   ❌ 可以不保存
    EPS_background_features.npz  ❌ 可以不保存
    """