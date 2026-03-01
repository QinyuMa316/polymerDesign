import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdmolops
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.error')   # 关闭 error 级别日志

from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from MolOpt.GATmodel import GATModel
from MolOpt.molDataLoader import MolDataset
from torch_geometric.data import DataLoader
import numpy as np
import os
import warnings
warnings.filterwarnings("ignore")

# =========================
#  功能基团 SMARTS（侧基锚定版）
# =========================
# 说明：
# 1) 这些 SMARTS 都包含一个“锚点原子”[*:1]，表示连接到主链/骨架的原子；
# 2) 匹配后我们会“只删除侧基部分”，保留锚点原子，从而避免破坏主链结构；
# 3) 匹配得到多处相同基团时，将逐一“单实例删除并预测”，最后对同一分子内同类基团贡献取平均。
# 4) 这些模式偏向“常见侧基”，若某些官能团位于主链内部（例如环系、主链羰基等）则通常不会被这些侧基模式命中。

### ✨MODIFIED/NEW ✨：使用“锚定侧基”的 SMARTS ###
fg_sidechain_smarts = {
    # 侧基基元（锚点为连接主链的原子；只删除侧基，不删锚点）
    "-OH":     "[*:1]-[OX2H]",
    "-NH2":    "[*:1]-[NX3;H2]",
    "-SH":     "[*:1]-[#16X2H]",
    # "-NO2":    "[$([*:1]-[NX3](=O)=O),$([*:1]-[N+](=O)[O-])]",
    "-CN":     "[*:1]-[CX2]#N",
    "-COOH":   "[*:1]-C(=O)[OH]",
    "-CHO":    "[*:1]-[CX3H1](=O)[#6]",
    "-F":      "[*:1]-[F]",
    "-Cl":     "[*:1]-[Cl]",
    "-Br":     "[*:1]-[Br]",
    "-I":      "[*:1]-[I]",

    # 简单不饱和侧基（严格作为支链，不匹配主链内烯/炔）
    "-C=CH2":  "[*:1]-C=[CH2]",
    "-C≡CH":   "[*:1]-C#[CH]",
}

fg_sidechain_smarts_add = {

    # 体积大、疏极性/可提高Tg - 烷基类/芳氧/烷氧芳基
    "-CH3": "[*:1]-[CH3]",
    "-CH(CH3)2":"[*:1]-C(-[CH3])-[CH3]",
    "-C(CH3)3": "[*:1]-C(-[CH3])(-[CH3])-[CH3]",
    "-cHex": "[*:1]-C1CCCCC1",

    # 芳环侧基（常见“苯环作侧基”情形；主链苯环不会命中该模式）
    "-Ph": "[*:1]-c1ccccc1",  # -Ar v6修改
    "-OPh":   "[*:1]-O-c1ccccc1",
    "-CH2Ph": "[*:1]-[CH2]-c1ccccc1",
    "-OCH2Ph":"[*:1]-O-[CH2]-c1ccccc1",
    "-PhPh":  "[*:1]-c1ccccc1-c2ccccc2",

    # 含氟侧基
    "-CF3":       "[*:1]-C(F)(F)F",
    "-OCF3":      "[*:1]-O-C(F)(F)F",
    "-C2F5":      "[*:1]-[C](F)(F)-[C](F)(F)F",
    "-OCF2CF3":   "[*:1]-O-[C](F)(F)-[C](F)(F)F",
    "-CF2CF2F3":  "[*:1]-[C](F)(F)-[C](F)(F)-[C](F)(F)F",
    "-OCF2CF2F3": "[*:1]-O-[C](F)(F)-[C](F)(F)-[C](F)(F)F",
    "-C6F5":      "[*:1]-c1c(F)c(F)c(F)c(F)c1F",
    # 强吸电子/可提高Tg（模型自行权衡对k的影响）
    "-SO2CF3": "[*:1]-S(=O)(=O)-C(F)(F)F",
    "-SO2Ph": "[*:1]-S(=O)(=O)-c1ccccc1",
    # 硅/硅氧烷侧基
    "-Si(CH3)3":    "[*:1]-[Si](-[CH3])(-[CH3])-[CH3]",
    "-OSi(CH3)3":   "[*:1]-O-[Si](-[CH3])(-[CH3])-[CH3]",
}

# # 刚性笼状基团
# "-Adamantyl": "[*:1]-C1C2CC3CC1CC(C2)C3",
# "-Norbornyl":  "[*:1]-C1C2CCC1C2",

fg_sidechain_smarts.update(fg_sidechain_smarts_add)

functional_groups = fg_sidechain_smarts  # ✨MODIFIED/NEW ✨：改为侧基锚定词典

log_file = 'attr.log'

# =============== 工具函数 ===============
def _safe_sanitize(mol):
    """尽量稳健地清理分子并返回副本。"""
    if mol is None:
        return None
    mol = Chem.Mol(mol)
    try:
        rdmolops.SanitizeMol(mol)
    except Exception:
        try:
            Chem.SanitizeMol(mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                       Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                       Chem.SanitizeFlags.SANITIZE_ADJUSTHS |
                                       Chem.SanitizeFlags.SANITIZE_CLEANUP |
                                       Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        except Exception:
            return None
    return mol


def is_pendant_sidechain(mol: Chem.Mol, match: tuple, anchor_qidx: int) -> bool:
    """
    返回 True 仅当：
      (A) 匹配到的功能团原子（除锚点外）没有与匹配外的原子相连（即“整团只通过锚点连到骨架”）
      (B) 锚点在原分子里与匹配外还至少有 1 个额外邻居（说明它属于骨架，不是末端单原子）
    """
    match_set = set(match)
    anchor_atom_idx = match[anchor_qidx]
    anchor_atom = mol.GetAtomWithIdx(anchor_atom_idx)

    # (A) 功能团原子（除锚点）不得与匹配外原子相连
    for i, aidx in enumerate(match):
        if i == anchor_qidx:
            continue
        atom = mol.GetAtomWithIdx(aidx)
        for nb in atom.GetNeighbors():
            nb_idx = nb.GetIdx()
            if nb_idx not in match_set:
                return False

    # (B) 锚点必须还有至少 1 个“匹配外”的邻居
    outside_neighbors = 0
    for nb in anchor_atom.GetNeighbors():
        if nb.GetIdx() not in match_set:
            outside_neighbors += 1
    if outside_neighbors < 1:
        return False

    return True


def get_sidechain_matches(mol, smarts):
    patt = Chem.MolFromSmarts(smarts)
    if patt is None:
        return None, [], None

    # 找到 SMARTS 查询里的锚点原子索引（[*:1]）
    anchor_qidx = None
    for ai in patt.GetAtoms():
        if ai.GetAtomMapNum() == 1:
            anchor_qidx = ai.GetIdx()
            break
    if anchor_qidx is None:
        anchor_qidx = 0  # 降级使用 0

    # 初步匹配
    raw_matches = mol.GetSubstructMatches(patt, uniquify=True)
    if not raw_matches:
        return patt, [], anchor_qidx

    # 仅保留“吊坠式侧基”匹配
    filtered = []
    for m in raw_matches:
        if is_pendant_sidechain(mol, m, anchor_qidx):
            filtered.append(m)

    return patt, filtered, anchor_qidx


def delete_one_fg_instance(mol, match, anchor_qidx):
    """删除匹配中的“非锚点原子”，保留锚点，实现剪掉侧基。"""
    if mol is None:
        return None
    match = list(match)
    anchor_atom_idx = match[anchor_qidx]
    _ = anchor_atom_idx  # 仅表明我们保留它
    to_delete = [idx for i, idx in enumerate(match) if i != anchor_qidx]

    em = Chem.EditableMol(Chem.Mol(mol))
    for aid in sorted(to_delete, reverse=True):
        try:
            em.RemoveAtom(aid)
        except Exception:
            return None
    newmol = em.GetMol()
    newmol = _safe_sanitize(newmol)
    return newmol


# ===== 预测函数 =====
def predict_property(model, smiles, device, mask_indices=None):
    """
    ✨说明：此版本不使用 mask；为兼容原框架，保留该签名但忽略 mask_indices。
    """
    try:
        dataset = MolDataset([smiles], [0])
        loader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                if not hasattr(batch, 'edge_index') or batch.edge_index.size(1) == 0:
                    return None
                out, _ = model(batch)
                return out.cpu().numpy()[0]
    except:
        return None


# ================ 多进程并行：按“分子”并行 ================
from multiprocessing import Pool
from functools import partial

_model_global = None
_device_global = None

def _init_worker(model_state_dict, device_str, num_layers=3, hidden_dim=128):
    """
    每个子进程初始化：重建模型、加载权重到本进程的 _model_global。
    """
    global _model_global, _device_global
    _device_global = torch.device(device_str)

    model = GATModel(num_node_features=7, num_edge_features=6,
                     hidden_dim=hidden_dim, num_heads=4, dropout=0.2,
                     num_layers=num_layers)
    model.load_state_dict(model_state_dict)
    model.to(_device_global)
    model.eval()
    _model_global = model


def _process_one_smiles(smiles: str):
    # _process_one_smiles(smiles: str, target=target):
    """
    单分子处理：基线预测 → 找 FG 吊坠匹配 → 逐实例删除并预测 → per-FG 均值。
    返回该 SMILES 的若干记录(list of dict)。
    """
    global _model_global, _device_global
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        mol = _safe_sanitize(mol)
        if mol is None or mol.GetNumAtoms() < 2:
            return []

        # 基线预测
        base_pred = predict_property(_model_global, smiles, _device_global)
        if base_pred is None:
            return []

        out_records = []
        # 针对每一个功能基团（仅限“侧基锚定”词典）
        for fg_name, smarts in functional_groups.items():
            patt, matches, anchor_qidx = get_sidechain_matches(mol, smarts)
            if not matches:
                continue
            n_matches_total = len(matches)
            # per_instance_contrib = []
            for inst_id, m in enumerate(matches):
                pruned = delete_one_fg_instance(mol, m, anchor_qidx)
                if pruned is None:
                    continue
                smi_pruned = Chem.MolToSmiles(pruned, canonical=True)
                pred_after = predict_property(_model_global, smi_pruned, _device_global)
                if pred_after is None:
                    continue
                contrib_i = float(base_pred - pred_after)
                # 每个实例单独一条记录
                out_records.append({
                    'fg': fg_name,
                    'attribution': contrib_i,         # ← 单实例贡献
                    'n_instances': n_matches_total,   # 该分子该FG的总匹配数（便于统计）
                    'instance_id': int(inst_id),      # 实例序号（0..n-1）
                    'smiles': smiles,
                    # 可选：如需调试可加入 'pruned_smiles': smi_pruned,
                })

            #     per_instance_contrib.append(contrib_i)
            #
            # if per_instance_contrib:
            #     avg_contrib = float(np.mean(per_instance_contrib))
            #     out_records.append({
            #         'fg': fg_name,
            #         'attribution': avg_contrib,        # per-instance 平均贡献（分子级）
            #         'n_instances': len(per_instance_contrib),
            #         'smiles': smiles,
            #     })
        return out_records

    except Exception as e:
        # 子进程里尽量不打印长错误，返回空让主进程汇总
        return []


def collect_fg_attributions_avg_delete_mp(df, model, device, target, num_workers=4):
    """
    多进程版本：将“逐分子归因”并行化。
    """
    device_str = str(device)
    model_state = model.state_dict()

    smiles_list = df['SMILES'].tolist()

    attribution_records = []
    with Pool(processes=num_workers,
              initializer=_init_worker,
              initargs=(model_state, device_str)) as pool:
        # 并行按分子处理
        # func = partial(_process_one_smiles, target=target)
        func = _process_one_smiles
        for rec_list in tqdm(pool.imap(func, smiles_list), total=len(smiles_list)):
            if rec_list:
                attribution_records.extend(rec_list)

    if attribution_records:
        return pd.DataFrame(attribution_records)
    else:
        print("Warning: No attribution data collected (avg-delete, mp)!")
        return pd.DataFrame(columns=['fg', 'attribution', 'n_instances', 'smiles'])



# =============== 绘图（原逻辑保留） ===============

# ———— v5修改：按 fg 分组，用 IQR rule 删除异常点（只影响统计与绘图，不影响原始归因CSV）
def remove_outliers_iqr_by_fg(df_attr, group_col='fg', value_col='attribution',
                              whisker=1.5, min_group_size=8):
    """
    对每个功能基团 fg：
      计算 Q1/Q3/IQR，并删除 attribution 落在 [Q1-1.5*IQR, Q3+1.5*IQR] 之外的点。
    为避免小样本被误伤：当该 fg 样本数 < min_group_size 时，不做删点。
    """
    df = df_attr.copy()
    keep_mask = np.ones(len(df), dtype=bool)

    for fg, sub in df.groupby(group_col):
        if len(sub) < min_group_size:
            continue  # 小样本不删

        q1 = sub[value_col].quantile(0.25)
        q3 = sub[value_col].quantile(0.75)
        iqr = q3 - q1

        # 若IQR为0（所有值相同或极其集中），无需删点
        if iqr == 0 or np.isnan(iqr):
            continue

        low = q1 - whisker * iqr
        high = q3 + whisker * iqr

        idx = sub.index
        keep_mask[idx] = (sub[value_col] >= low) & (sub[value_col] <= high)

    return df.loc[keep_mask].reset_index(drop=True)


def plot_violin_filter(df_attr, property_name, out_dir,
                       agg_type="mean",
                       sort=False, min_count=5):
    # plt.rcParams.update({'font.size': 11})
    plt.rcParams.update({
        'font.family': 'Arial',
        'axes.titlesize': 14,  # 主标题
        'axes.labelsize': 14,  # 横纵轴标题
        'xtick.labelsize': 12,  # x刻度
        'ytick.labelsize': 12,  # y刻度
        'legend.fontsize': 12   #  图例字号
    })
    fg_counts = df_attr['fg'].value_counts()
    valid_fgs = fg_counts[fg_counts >= min_count].index
    df_attr = df_attr[df_attr['fg'].isin(valid_fgs)].copy()

    # fg_means = df_attr.groupby('fg')['attribution'].mean()
    # ———— v3修改
    if agg_type == "median":
        fg_stats = df_attr.groupby('fg')['attribution'].median()
    else:
        fg_stats = df_attr.groupby('fg')['attribution'].mean()
    # name_map = {fg: f'{fg} ({fg_means[fg]:.2f})' for fg in valid_fgs}
    name_map = {fg: f'{fg} ({fg_stats[fg]:.2f})' for fg in valid_fgs}
    # ———— v3修改
    df_attr['fg_label'] = df_attr['fg'].map(name_map)

    # ———— v2修改
    master_order_fg = [fg for fg in functional_groups.keys() if fg in valid_fgs]
    master_order_labels = [name_map[fg] for fg in master_order_fg]
    # ———— v2修改
    if sort:
        # ———— v3修改
        if agg_type == "median":
            order = df_attr.groupby('fg_label')['attribution'].median().sort_values().index
        elif agg_type == "mean":
            order = df_attr.groupby('fg_label')['attribution'].mean().sort_values().index
        png_suffix = f'sorted_{agg_type}'
        # ———— v3修改
    else:
        # order = df_attr['fg_label'].unique()
        order = master_order_labels
        png_suffix = 'unsorted'

    plt.figure(figsize=(6.5, max(6, len(order) * 0.45)))
    sns.violinplot(data=df_attr, x='attribution', y='fg_label',
                   palette='coolwarm', inner='box', order=order)
    # todo: 换色 palette=
    plt.axvline(0, color='k', linestyle='--')
    if property_name == 'EPS':
        property_name = 'k'
    plt.title(f'Functional Group Attribution Distribution for {property_name}')
    plt.xlabel('Attribution')
    plt.ylabel('Functional Groups')
    plt.tick_params(axis='both', labelsize=11)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/attribution_violin_{property_name}_{png_suffix}.png', dpi=600)


def plot_scatter_filter(df_attr, property_name, out_dir,
                        agg_type="mean",
                        sort=True, min_count=5):
    plt.rcParams.update({'font.size': 11})
    fg_counts = df_attr['fg'].value_counts()
    valid_fgs = fg_counts[fg_counts >= min_count].index
    df_attr = df_attr[df_attr['fg'].isin(valid_fgs)].copy()

    # fg_means = df_attr.groupby('fg')['attribution'].mean()
    # ———— v3修改
    if agg_type == "median":
        fg_stats = df_attr.groupby('fg')['attribution'].median()
    else:
        fg_stats = df_attr.groupby('fg')['attribution'].mean()
    # name_map = {fg: f'{fg} ({fg_means[fg]:.2f})' for fg in valid_fgs}
    name_map = {fg: f'{fg} ({fg_stats[fg]:.2f})' for fg in valid_fgs}
    # ———— v3修改

    df_attr['fg_label'] = df_attr['fg'].map(name_map)

    # ———— v2修改
    master_order_fg = [fg for fg in functional_groups.keys() if fg in valid_fgs]
    master_order_labels = [name_map[fg] for fg in master_order_fg]
    # ———— v2修改
    if sort:
        # ———— v3修改
        if agg_type == "median":
            order = df_attr.groupby('fg_label')['attribution'].median().sort_values().index
        elif agg_type == "mean":
            order = df_attr.groupby('fg_label')['attribution'].mean().sort_values().index
        png_suffix = f'sorted_{agg_type}'
        # ———— v3修改
    else:
        # order = df_attr['fg_label'].unique()
        order = master_order_labels
        png_suffix = 'unsorted'

    fig, ax = plt.subplots(figsize=(6.5, max(6, len(order) * 0.45)))
    red = '#7A9EBA'
    blue = '#D28C8C'

    for i, fg in enumerate(order):
        fg_data = df_attr[df_attr['fg_label'] == fg]['attribution']
        jitter = np.random.uniform(-0.2, 0.2, size=len(fg_data))
        y_pos = np.full(len(fg_data), i) + jitter
        colors = [red if x > 0 else blue for x in fg_data]
        plt.scatter(fg_data, y_pos, facecolors='none', edgecolors=colors,
                    linewidth=1, s=50, alpha=0.7)
    if property_name == 'EPS':
        property_name = 'k'
    plt.axvline(0, color='k', linestyle='--')
    plt.yticks(range(len(order)), order)
    plt.title(f'Functional Group Attribution for {property_name}')
    plt.xlabel('Attribution')
    plt.ylabel('Functional Groups')
    plt.tick_params(axis='both', labelsize=11)
    fig.subplots_adjust(top=0.96, bottom=0.04)
    plt.tight_layout()
    plt.savefig(f'{out_dir}/attribution_scatter_{property_name}_{png_suffix}.png', dpi=600)


if __name__ == '__main__':
    agg_type = "median"
    clean_outliers = True

    n_core = 4
    # note: test version 1k data only
    data_path = 'data/PI1M_v2.csv'
    attrfg_dir = 'attrfg1m'  # without NO2
    os.makedirs(attrfg_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_layers = 3
    hidden_dim = 128
    model = GATModel(num_node_features=7, num_edge_features=6, hidden_dim=hidden_dim,
                     num_heads=4, dropout=0.2, num_layers=num_layers)
    pretrain_epo = 50

    df = pd.read_csv(data_path)

    fg_results = {}

    for target in ['EPS', 'Tg']:
        csv_path = f'{attrfg_dir}/attribution_{target}.csv'
        # ⭐️【贡献列表未计算，开始计算】
        if not os.path.exists(csv_path):
            print(f"🚀Processing {target} attribution...")
            model_dir = 'models_polymer'
            model_path = f'{model_dir}/best_model_{target}_cl_{pretrain_epo}epo_aug12.pth'
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                model.to(device)
                model.eval()
                print(f'Successfully loaded model from {model_path}')
            else:
                print(f'Failed to load model from {model_path}')

            # 并行配置
            core_count = os.cpu_count()
            print(f"CPU cores: {core_count}, 使用 {n_core} 个进程并行")

            # ✨ 并行的“删除侧基 + 实例平均”收集函数
            df_attr = collect_fg_attributions_avg_delete_mp(
                df, model, device, target, num_workers=n_core
            )

            if not df_attr.empty:
                df_attr.to_csv(csv_path, index=False)
            else:
                print(f"Warning: No attribution data for {target}, CSV not created!")
                continue
        # ⭐️【贡献列表已计算，直接读取，开始绘图】
        else:
            print(f'✅{target} attribution already processed.')
            df_attr = pd.read_csv(csv_path)

        if not df_attr.empty and 'fg' in df_attr.columns:
            # if agg_type == "mean":
            #     grouped = df_attr.groupby('fg')['attribution'].mean()
            # elif agg_type == "median":
            #     grouped = df_attr.groupby('fg')['attribution'].mediam()
            # ———— v5修改
            if clean_outliers:
                df_attr = remove_outliers_iqr_by_fg(
                    df_attr,
                    group_col='fg',
                    value_col='attribution',
                    whisker=1.5,
                    min_group_size=8
                )
                grp = df_attr.groupby('fg')['attribution']
            else:
                grp = df_attr.groupby('fg')['attribution']
            # ———— v5修改

            # ———— v4修改
            mean_vals = grp.mean()
            median_vals = grp.median()
            q1_vals = grp.quantile(0.25)
            q3_vals = grp.quantile(0.75)
            iqr_vals = q3_vals - q1_vals

            for fg, value in mean_vals.items():
                # mean_vals.index == median_vals.index == q1_vals.index == q3_vals.index
                # 代表：所有在 df_attr 里至少出现过一次的功能基团列表
                if fg not in fg_results:
                    fg_results[fg] = {}
                prop_name = 'k' if target == 'EPS' else target # EPS → k 的重命名

                # fg_results[fg][target] = round(value, 2)
                fg_results[fg][f'{prop_name}_mean'] = round(mean_vals[fg], 2)
                fg_results[fg][f'{prop_name}_median'] = round(median_vals[fg], 2)
                fg_results[fg][f'{prop_name}_IQR'] = round(iqr_vals[fg], 2)
            # ———— v4修改

            # [sorted]
            plot_violin_filter(df_attr, property_name=target, out_dir = attrfg_dir,
                               agg_type=agg_type,
                               sort = True, min_count = 5)
            plot_scatter_filter(df_attr, property_name=target, out_dir = attrfg_dir,
                                agg_type=agg_type,
                                sort = True, min_count=5)
            # [unsorted]
            plot_violin_filter(df_attr, property_name=target, out_dir = attrfg_dir,
                               agg_type=agg_type,
                               sort = False, min_count = 5)
            plot_scatter_filter(df_attr, property_name=target, out_dir = attrfg_dir,
                                agg_type=agg_type,
                                sort = False, min_count=5)
        else:
            print(f"Warning: {csv_path} is empty!")
    # ⭐️【汇总基团贡献表格：基团-对k贡献-对Tg贡献】
    fg_results_path = f'{attrfg_dir}/groups_attr.csv'
    if fg_results and not os.path.exists(fg_results_path):
        result_df = pd.DataFrame.from_dict(fg_results, orient='index')

        # ———— v2修改
        ordered_index = [fg for fg in functional_groups.keys() if fg in result_df.index]
        result_df = result_df.reindex(ordered_index)
        # ———— v2修改

        result_df.index.name = 'functional groups'
        # result_df = result_df[['EPS', 'Tg']] if 'EPS' in result_df.columns else result_df
        # ———— v4修改
        ordered_cols = [
            'k_mean', 'k_median', 'k_IQR',
            'Tg_mean', 'Tg_median', 'Tg_IQR'
        ]
        existing_cols = [c for c in ordered_cols if c in result_df.columns]
        result_df = result_df[existing_cols]
        # ———— v4修改
        result_df.index = result_df.index.map(
            lambda x: f"'{x}" if any(c in x for c in ['=', '-', '+', '<', '>']) else x)
        result_df.to_csv(fg_results_path, encoding='utf-8-sig')
        print("功能基团贡献表已保存。")
    elif not fg_results:
        print("没有可用的功能基团贡献数据。")
    elif os.path.exists(fg_results_path):
        print("功能基团贡献表已存在。")

    # v2 排序按照 functional_groups 进行排序
    #  plot_scatter_filter：同样按 functional_groups 顺序
    #  groups_attr.csv 汇总表：按 functional_groups 顺序重排行
    # hyperparam_v2_1_22 设定一个参数（string类型），对于plot_violin_filter和groups_attr.csv，可以选择是要均值还是中位数
    # v4 groups_attr.csv升级同时保存：mean + median + IQR
    """
    # note:
    #  在 violin plot（小提琴图） 中，标注的中位线表示的是“中位数（median）”，不是均值（mean）。
    IQR 是 Interquartile Range，中文叫四分位距。它不是一个“中心值”，而是一个衡量数据分布离散程度、稳定性和不确定性的统计量
    对一组数值按从小到大排序后：Q1（第一四分位数）：25% 分位点；Q3（第三四分位数）：75% 分位点
    IQR = Q3 − Q1，也就是说，IQR 描述的是：中间 50% 数据所覆盖的区间宽度。
    你现在的 attribution 数据有三个显著特点：
    一是样本来自不同骨架、不同取代环境；
    二是存在长尾（极端正贡献或极端负贡献）；
    三是你已经意识到均值会被“少数极端体系”拖偏。
    在这种情况下：中位数告诉你“典型情况下这个基团是升还是降”，IQR告诉你“这个判断有多稳”
    另外，IQR 不会被极端值支配，这一点和中位数一样。
    所以它比标准差更适合你这种“非高斯、长尾、混合分布”的归因结果。
    """
    # v5 删除异常值再进行统计 // scatter 上保留异常值，violin删除异常值，groups_attr.csv删除异常值
    # v6 对functional group修改一些名称&顺序
    # v7
    # todo: 换色 palette=

