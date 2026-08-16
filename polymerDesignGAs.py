import os
from pathlib import Path
import random
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import KekulizeException
import torch
from torch_geometric.data import DataLoader
from MolOpt.molDataLoader import MolDataset
from MolOpt.GATmodel import GATModel
from MolOpt.plotGenMols import PlotGenMols, visualize_base_polymers
from MolOpt.calSAScore import calSAScore
from MolGA.designPolyGA import GeneticAlgorithm
from MolGA.designPolyNSGA2 import NSGA2

# ========= 0. Tg 预测模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATModel(num_node_features=7, num_edge_features=6,
                 hidden_dim=128, num_heads=4,
                 dropout=0.2, num_layers=3)

def predict_property(smiles: str, target="Tg") -> float:
    suffix = "_cl_50epo_aug12"
    mdl_path = f"models_polymer/best_model_{target}{suffix}.pth"
    chkpt = torch.load(mdl_path, map_location=device)
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device).eval()

    ds  = MolDataset([smiles], [0])
    dl  = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            out, _ = model(batch.to(device))
            return float(out.cpu().numpy()[0])

frag_smiles = [
    # ===== 基础侧基（对应 fg_sidechain_smarts）=====
    "C",                # -CH3
    "C(F)(F)F",         # -CF3
    "O",                # -OH
    "[NH2]",            # -NH2
    "S",                # -SH  （二价硫；连接后为 -S-）
    "[N+](=O)[O-]",     # -NO2
    "C#N",              # -CN
    "C(=O)O",           # -COOH
    "C=O",              # -CHO
    "F",                # -F
    "Cl",               # -Cl
    "Br",               # -Br
    "I",                # -I

    "C=C",              # -C=CH2   （base–CH=CH2）
    "C#C",              # -C≡CH    （base–C≡CH）

    "c1ccccc1",         # -Ar（苯基）

    # ===== 扩展侧基（对应 fg_sidechain_smarts_add）=====
    # 体积大/疏极性
    "[CH](C)C",         # -CHMe2   （异丙基，锚点为中心CH）
    "C(C)(C)C",         # -CMe3    （叔丁基，锚点为中心C）
    "C1CCCCC1",         # -cHex    （环己基）

    "Oc1ccccc1",        # -OPh     （芳氧）
    "Cc1ccccc1",        # -CH2Ph   （苄基）
    "OCc1ccccc1",       # -OCH2Ph  （苄氧基）
    "c1ccccc1c2ccccc2", # -BiPh    （联苯基）

    # 含氟侧基
    "OC(F)(F)F",                        # -OCF3
    "C(F)(F)C(F)(F)F",                  # -C2F5（= CF2CF3）
    "OC(F)(F)C(F)(F)F",                 # -OCF2CF3
    "C(F)(F)C(F)(F)C(F)(F)F",           # -CF2CF2CF3
    "OC(F)(F)C(F)(F)C(F)(F)F",          # -OCF2CF2CF3
    "c1c(F)c(F)c(F)c(F)c1F",            # -C6F5（五氟苯基，锚接于未氟化位）

    # 硅/硅氧烷
    "[Si](C)(C)C",      # -SiMe3
    "O[Si](C)(C)C",     # -OSiMe3

    # 强吸电子
    "S(=O)(=O)c1ccccc1",# -SO2Ph
    "S(=O)(=O)C(F)(F)F" # -SO2CF3
]


# ===【计算评价指标】==== 开始

def set_global_seed(seed: int):
    """Set random seeds for reproducible GA/NSGA-II runs."""
    random.seed(seed)
    try:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _valid_objective_df(df: pd.DataFrame) -> pd.DataFrame:
    """Keep valid molecules and remove duplicated SMILES before metric calculation."""
    required = {"smiles", "eps", "tg"}
    if df is None or df.empty or not required.issubset(df.columns):
        return pd.DataFrame(columns=["smiles", "eps", "tg"])
    out = df.dropna(subset=["smiles", "eps", "tg"]).copy()
    out = out.drop_duplicates(subset=["smiles"], keep="first")
    out["eps"] = pd.to_numeric(out["eps"], errors="coerce")
    out["tg"] = pd.to_numeric(out["tg"], errors="coerce")
    out = out.dropna(subset=["eps", "tg"])
    return out


def _to_minimized_normalized_points(df: pd.DataFrame,
                                    eps_min: float = 2.0,
                                    eps_max: float = 4.0,
                                    tg_min: float = 400.0,
                                    tg_max: float = 800.0):
    """
    Convert (eps, tg) to normalized minimization objectives.
    Objective 1: normalized eps, lower is better.
    Objective 2: normalized -tg, lower is better, implemented as (tg_max - tg)/(tg_max - tg_min).
    """
    df = _valid_objective_df(df)
    points = []
    for _, row in df.iterrows():
        eps = float(row["eps"])
        tg = float(row["tg"])
        x = (eps - eps_min) / (eps_max - eps_min) if eps_max != eps_min else 0.0
        y = (tg_max - tg) / (tg_max - tg_min) if tg_max != tg_min else 0.0
        # clip to reference box for stable HV calculation
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(1.0, y))
        points.append((x, y))
    return points


def _dominates(a, b, eps: float = 1e-12) -> bool:
    """Return True if point a strictly dominates point b in a minimization problem."""
    return all(x <= y + eps for x, y in zip(a, b)) and any(x < y - eps for x, y in zip(a, b))


def _nondominated_points(points):
    """Return nondominated points for a minimization problem."""
    pts = list(dict.fromkeys(points))
    nd = []
    for i, p in enumerate(pts):
        dominated = False
        for j, q in enumerate(pts):
            if i != j and _dominates(q, p):
                dominated = True
                break
        if not dominated:
            nd.append(p)
    return nd


def _hypervolume_2d(points, reference=(1.0, 1.0)) -> float:
    """Calculate 2D hypervolume of nondominated minimization points within the reference box."""
    nd = _nondominated_points(points)
    nd = [(x, y) for x, y in nd if x <= reference[0] and y <= reference[1]]
    if not nd:
        return 0.0
    nd = sorted(nd, key=lambda p: (p[0], p[1]))
    hv = 0.0
    prev_y = reference[1]
    for x, y in nd:
        if y < prev_y:
            hv += max(0.0, reference[0] - x) * max(0.0, prev_y - y)
            prev_y = y
    return hv


def _spacing(points) -> float:
    """
    Calculate spacing of the nondominated points.
    Lower spacing indicates a more evenly distributed Pareto set.
    """
    nd = _nondominated_points(points)
    if len(nd) < 2:
        return 0.0
    nearest_distances = []
    for i, p in enumerate(nd):
        dmin = min(abs(p[0] - q[0]) + abs(p[1] - q[1])
                   for j, q in enumerate(nd) if i != j)
        nearest_distances.append(dmin)
    mean_d = sum(nearest_distances) / len(nearest_distances)
    if len(nearest_distances) < 2:
        return 0.0
    var = sum((d - mean_d) ** 2 for d in nearest_distances) / (len(nearest_distances) - 1)
    return var ** 0.5


def _coverage(points_a, points_b) -> float:
    """C(A, B): fraction of points in B dominated by at least one point in A."""
    nd_a = _nondominated_points(points_a)
    nd_b = _nondominated_points(points_b)
    if not nd_b:
        return 0.0
    dominated_count = 0
    for b in nd_b:
        if any(_dominates(a, b) for a in nd_a):
            dominated_count += 1
    return dominated_count / len(nd_b)


def _load_jsonl_population(jsonl_path: str) -> pd.DataFrame:
    """Load GA/NSGA-II population records saved as jsonl."""
    rows = []
    if not os.path.exists(jsonl_path):
        return pd.DataFrame()
    with open(jsonl_path, "r", encoding="utf-8") as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return pd.DataFrame(rows)


def _generation_metric_table(ga_df: pd.DataFrame, nsga_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate HV, Pareto coverage, and spacing for each generation."""
    gens = sorted(set(ga_df.get("generation", pd.Series(dtype=int)).dropna().astype(int).tolist()) |
                  set(nsga_df.get("generation", pd.Series(dtype=int)).dropna().astype(int).tolist()))
    records = []
    for gen in gens:
        ga_gen = ga_df[ga_df["generation"].astype(int) == gen] if not ga_df.empty else pd.DataFrame()
        nsga_gen = nsga_df[nsga_df["generation"].astype(int) == gen] if not nsga_df.empty else pd.DataFrame()
        ga_points = _to_minimized_normalized_points(ga_gen)
        nsga_points = _to_minimized_normalized_points(nsga_gen)
        records.append({
            "generation": gen,
            "GA_hypervolume": _hypervolume_2d(ga_points),
            "NSGA2_hypervolume": _hypervolume_2d(nsga_points),
            "GA_spacing": _spacing(ga_points),
            "NSGA2_spacing": _spacing(nsga_points),
            "C_NSGA2_GA": _coverage(nsga_points, ga_points),
            "C_GA_NSGA2": _coverage(ga_points, nsga_points),
            "GA_front_size": len(_nondominated_points(ga_points)),
            "NSGA2_front_size": len(_nondominated_points(nsga_points)),
        })
    return pd.DataFrame(records)


def _plot_metric_trends(metric_df: pd.DataFrame, out_dir: str):
    """Plot per-generation trends of hypervolume, Pareto coverage, and spacing."""
    if metric_df.empty:
        return
    os.makedirs(out_dir, exist_ok=True)

    # =====【折线图转散点图】===== 开始
    # Hypervolume：使用有色空心圆圈散点，不再连接相邻 generation
    plt.figure(figsize=(6, 4))
    plt.scatter(metric_df["generation"], metric_df["GA_hypervolume"],
                label="GA", marker="o", s=42,
                facecolors="none", edgecolors="tab:blue", linewidths=1.5)
    plt.scatter(metric_df["generation"], metric_df["NSGA2_hypervolume"],
                label="NSGA-II", marker="o", s=42,
                facecolors="none", edgecolors="tab:orange", linewidths=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume")
    plt.title("Hypervolume Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/metric_hypervolume_trend.png", dpi=600)
    plt.close()

    # Pareto coverage：两组指标分别使用不同颜色的空心圆圈
    plt.figure(figsize=(6, 4))
    plt.scatter(metric_df["generation"], metric_df["C_NSGA2_GA"],
                label=r"$C$(NSGA-II, GA)", marker="o", s=42,
                facecolors="none", edgecolors="tab:green", linewidths=1.5)
    plt.scatter(metric_df["generation"], metric_df["C_GA_NSGA2"],
                label=r"$C$(GA, NSGA-II)", marker="o", s=42,
                facecolors="none", edgecolors="tab:red", linewidths=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Pareto Coverage")
    plt.title("Pareto Coverage Evolution")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/metric_pareto_coverage_trend.png", dpi=600)
    plt.close()

    # Spacing：使用有色空心圆圈散点，不再绘制折线
    plt.figure(figsize=(6, 4))
    plt.scatter(metric_df["generation"], metric_df["GA_spacing"],
                label="GA", marker="o", s=42,
                facecolors="none", edgecolors="tab:purple", linewidths=1.5)
    plt.scatter(metric_df["generation"], metric_df["NSGA2_spacing"],
                label="NSGA-II", marker="o", s=42,
                facecolors="none", edgecolors="tab:brown", linewidths=1.5)
    plt.xlabel("Generation")
    plt.ylabel("Spacing")
    plt.title("Spacing Evolution")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{out_dir}/metric_spacing_trend.png", dpi=600)
    plt.close()
    #=====【折线图转散点图】===== 结束


def compare_ga_nsga2_metrics(ga_dir: str, nsga_dir: str, out_dir: str) -> dict:
    """
    Post-process GA and NSGA-II jsonl outputs and compare final nondominated sets.
    Also saves per-generation metric trends.
    """
    os.makedirs(out_dir, exist_ok=True)
    ga_df = _load_jsonl_population(f"{ga_dir}/ga_population.jsonl")
    nsga_df = _load_jsonl_population(f"{nsga_dir}/ga_population.jsonl")

    metric_df = _generation_metric_table(ga_df, nsga_df)
    metric_df.to_csv(f"{out_dir}/metrics_by_generation.csv", index=False)
    _plot_metric_trends(metric_df, out_dir)

    # Final metrics are calculated from the last recorded generation for both methods.
    if metric_df.empty:
        summary = {
            "GA_hypervolume": 0.0,
            "NSGA2_hypervolume": 0.0,
            "GA_spacing": 0.0,
            "NSGA2_spacing": 0.0,
            "C_NSGA2_GA": 0.0,
            "C_GA_NSGA2": 0.0,
            "GA_front_size": 0,
            "NSGA2_front_size": 0,
        }
    else:
        last = metric_df.iloc[-1].to_dict()
        summary = {k: last[k] for k in last if k != "generation"}
        summary["generation"] = int(last["generation"])

    pd.DataFrame([summary]).to_csv(f"{out_dir}/metrics_final_summary.csv", index=False)
    return summary

# ===【计算评价指标】==== 结束


if __name__ == "__main__":

    N_GEN = 50
    POP_SIZE = 40
    RANDOM_SEED = 42
    # RANDOM_SEED = random.randint(1, 10000)
    RUN_COMPARE_GA_NSGA2 = True
    print(f'RANDOM_SEED: {RANDOM_SEED}')

    # ========= 1. 核心骨架 & 替换位点 =========
    df = pd.read_csv('polyDesign_data/data_modification/polyimides_pred_prop_base_str_std.csv')
    scaffolds = df['scaffold'].tolist()
    scaffolds_unique = list(dict.fromkeys(scaffolds))
    print(len(scaffolds), "->", len(scaffolds_unique))

    base_polymers = []
    opt_results = []

    if RUN_COMPARE_GA_NSGA2:
        main_dir = f'resultCompareSeed{RANDOM_SEED}'
    else:
        version = 'NSGA-II'
        main_dir = f'result{version}'

    for id, scaffold in tqdm(enumerate(scaffolds_unique)):
        print('\n'+'='*50)
        print(f'id: {id}, Base polymer: {scaffold}')
        base_smi = scaffold
        base_tg = predict_property(base_smi, 'Tg')
        base_eps = predict_property(base_smi, 'EPS')
        base_sa = calSAScore(base_smi)
        base_polymers.append({
            "id": id,
            "smiles": scaffold,
            "tg": base_tg,
            "eps": base_eps,
            "sa": base_sa,
        })

        if RUN_COMPARE_GA_NSGA2:
            # ========= 在同一随机种子下分别运行 GA 和 NSGA-II =========
            ga_dir = f'{main_dir}/GA_id{id}'
            nsga_dir = f'{main_dir}/NSGA-II_id{id}'

            print('Running GA ...')
            set_global_seed(RANDOM_SEED)
            ga_model = GeneticAlgorithm(base_smi,
                                        base_tg, base_eps,
                                        frag_smiles, ga_dir,
                                        N_GEN=N_GEN, POP_SIZE=POP_SIZE)
            best_smi, best_prop = ga_model.run_ga(plot=True)
            best_eps, best_tg = best_prop
            print("GA Best SMILES :", best_smi)
            print(f'GA Best Tg: {best_tg:.1f} K, k: {best_eps:.3f}')

            print('Running NSGA-II ...')
            set_global_seed(RANDOM_SEED)
            nsga_model = NSGA2(base_smi,
                               base_tg, base_eps,
                               frag_smiles, nsga_dir,
                               N_GEN=N_GEN, POP_SIZE=POP_SIZE)
            pareto_solutions = nsga_model.run_ga(plot=True)
            frontier_size = len(pareto_solutions)
            print(f"NSGA-II Frontier size = {frontier_size:d}")

            # ====【opt result加优化后的结果】==== 开始
            # GA 只有一个最终 best solution，因此这里统计 best solution 相对于 base polymer 的变化。
            ga_sa = calSAScore(best_smi) if best_smi is not None else None
            ga_delta_tg = (best_tg - base_tg) / base_tg * 100 if base_tg != 0 else None
            # 沿用原代码定义：delta_eps = (optimized_eps - base_eps) / base_eps * 100
            # 因此 k 降低时该值为负数。若希望“k decrease”为正数，可改为 (base_eps - best_eps) / base_eps * 100。
            ga_delta_eps = (best_eps - base_eps) / base_eps * 100 if base_eps != 0 else None
            ga_delta_sa = ga_sa - base_sa if ga_sa is not None else None

            # NSGA-II 返回 Pareto frontier，因此统计 frontier 中所有解的平均变化。
            nsga_delta_tg_list = []
            nsga_delta_eps_list = []
            nsga_sa_list = []

            for sol in pareto_solutions:
                cur_tg = sol["tg"]
                cur_eps = sol["eps"]
                cur_smi = sol["smiles"]
                cur_sa = calSAScore(cur_smi)

                cur_delta_tg = (cur_tg - base_tg) / base_tg * 100 if base_tg != 0 else 0.0
                # 沿用原代码定义：cur_delta_eps = (cur_eps - base_eps) / base_eps * 100
                cur_delta_eps = (cur_eps - base_eps) / base_eps * 100 if base_eps != 0 else 0.0

                nsga_delta_tg_list.append(cur_delta_tg)
                nsga_delta_eps_list.append(cur_delta_eps)
                nsga_sa_list.append(cur_sa)

            if len(nsga_delta_tg_list) > 0:
                nsga_avg_tg = sum(nsga_delta_tg_list) / len(nsga_delta_tg_list)
                nsga_avg_eps = sum(nsga_delta_eps_list) / len(nsga_delta_eps_list)
                nsga_avg_sa = sum(nsga_sa_list) / len(nsga_sa_list)
                nsga_delta_avg_sa = nsga_avg_sa - base_sa
            else:
                nsga_avg_tg = None
                nsga_avg_eps = None
                nsga_avg_sa = None
                nsga_delta_avg_sa = None

            print(f"Base polymer Tg: {base_tg:.1f} K, k: {base_eps:.3f}, SAScore: {base_sa:.2f}")
            print(f"GA Tg change: {ga_delta_tg:.2f}%" if ga_delta_tg is not None else "GA Tg change: None")
            print(f"GA k change: {ga_delta_eps:.2f}%" if ga_delta_eps is not None else "GA k change: None")
            print(f"GA SAScore: {ga_sa:.2f}, Delta SAScore: {ga_delta_sa:.2f}" if ga_sa is not None else "GA SAScore: None")
            print(f"NSGA-II Avg Tg change: {nsga_avg_tg:.2f}%" if nsga_avg_tg is not None else "NSGA-II Avg Tg change: None")
            print(f"NSGA-II Avg k change: {nsga_avg_eps:.2f}%" if nsga_avg_eps is not None else "NSGA-II Avg k change: None")
            print(f"NSGA-II Avg SAScore: {nsga_avg_sa:.2f}, Delta Avg SAScore: {nsga_delta_avg_sa:.2f}" if nsga_avg_sa is not None else "NSGA-II Avg SAScore: None")
            # ====【opt result加优化后的结果】==== 结束

            # ===【计算评价指标】==== 开始
            metrics_dir = f'{main_dir}/metrics_id{id}'
            metric_summary = compare_ga_nsga2_metrics(ga_dir, nsga_dir, metrics_dir)
            print("=== Multi-objective metric comparison on final nondominated sets ===")
            for key, value in metric_summary.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            # ===【计算评价指标】==== 结束

            # ====【opt result加优化后的结果】==== 开始
            opt_results.append({
                "Base Polymer SMILES": base_smi,
                "Tg": round(base_tg, 1),
                "k": round(base_eps, 3),
                "SAScore": round(base_sa, 2),

                "GA Tg change (%)": round(ga_delta_tg, 2) if ga_delta_tg is not None else None,
                "GA k change (%)": round(ga_delta_eps, 2) if ga_delta_eps is not None else None,
                "GA SAScore": round(ga_sa, 2) if ga_sa is not None else None,
                "GA Delta SAScore": round(ga_delta_sa, 2) if ga_delta_sa is not None else None,

                "NSGA-II Frontier Size": frontier_size,
                "NSGA-II Avg Tg change (%)": round(nsga_avg_tg, 2) if nsga_avg_tg is not None else None,
                "NSGA-II Avg k change (%)": round(nsga_avg_eps, 2) if nsga_avg_eps is not None else None,
                "NSGA-II Avg SAScore": round(nsga_avg_sa, 2) if nsga_avg_sa is not None else None,
                "NSGA-II Avg Delta SAScore": round(nsga_delta_avg_sa, 2) if nsga_delta_avg_sa is not None else None,

                "GA HV": round(metric_summary.get("GA_hypervolume", 0.0), 4),
                "NSGA-II HV": round(metric_summary.get("NSGA2_hypervolume", 0.0), 4),
                "GA Spacing": round(metric_summary.get("GA_spacing", 0.0), 4),
                "NSGA-II Spacing": round(metric_summary.get("NSGA2_spacing", 0.0), 4),
                "C(NSGA-II, GA)": round(metric_summary.get("C_NSGA2_GA", 0.0), 4),
                "C(GA, NSGA-II)": round(metric_summary.get("C_GA_NSGA2", 0.0), 4),
                "GA Frontier Size": int(metric_summary.get("GA_front_size", 0)),
                "NSGA-II Metric Frontier Size": int(metric_summary.get("NSGA2_front_size", 0)),
            })
            # ====【opt result加优化后的结果】==== 结束

            # 原有分子可视化功能分别对两个结果目录执行
            for cur_version, cur_dir in [("GA", ga_dir), ("NSGA-II", nsga_dir)]:
                plotgm = PlotGenMols(base_eps, base_tg, cur_dir)
                plotgm.plotKDEbyGenerations('tg')
                plotgm.plotKDEbyGenerations('eps')
                plotgm.plotScatterbyGenerations()
                plotgm.plotScatterWithKDEbyGenerations(id=id, version=cur_version)
                plotgm.visualize_gen_bests(mols_per_row=4)
                if 'NS' in cur_version:
                    plotgm.visualize_pareto_frontier(mols_per_row=5)
            continue

        # ========= 原有单算法运行逻辑，保留备用 =========
        dir_folder = f'{main_dir}/{version}_id{id}'
        if 'NS' not in version:
            print('Running GA ...')
            GA = GeneticAlgorithm(base_smi,
                                  base_tg, base_eps,
                                  frag_smiles, dir_folder,
                                  N_GEN=N_GEN, POP_SIZE=POP_SIZE)
            best_smi, best_prop = GA.run_ga()
            best_eps, best_tg = best_prop
            print("Best SMILES :", best_smi)
            print(f'Best Tg: {best_tg:.1f} K, k: {best_eps:.3f}')

        elif 'NS' in version:
            print('Running NSGA-II ...')
            GA = NSGA2(base_smi, base_tg, base_eps,
                       frag_smiles, dir_folder, N_GEN=N_GEN, POP_SIZE=POP_SIZE)
            pareto_solutions = GA.run_ga(plot=True)
            frontier_size = len(pareto_solutions)

            delta_tg_list = []
            delta_eps_list = []
            sa_list = []
            for i, sol in enumerate(pareto_solutions):
                cur_tg = sol["tg"]
                cur_eps = sol["eps"]
                cur_smi = sol["smiles"]
                cur_sa = calSAScore(cur_smi)
                delta_tg = (cur_tg - base_tg) / base_tg * 100
                delta_eps = (cur_eps - base_eps) / base_eps * 100
                delta_tg_list.append(delta_tg)
                delta_eps_list.append(delta_eps)
                sa_list.append(cur_sa)
            avg_tg = sum(delta_tg_list) / len(delta_tg_list)
            avg_eps = sum(delta_eps_list) / len(delta_eps_list)
            avg_sa = sum(sa_list) / len(sa_list)
            delta_avg_sa = avg_sa - base_sa

            print(f"Base polymer Tg: {base_tg:.1f} K, k: {base_eps:.3f}, SAScore: {base_sa:.2f}")
            print(f"Frontier size = {frontier_size}")
            print(f"Avg Tg change: {avg_tg:.2f}%")
            print(f"Avg k change: {avg_eps:.2f}%")
            print(f"Avg SAScore: {avg_sa:.2f}, Delta Avg SAScore: {delta_avg_sa:.2f}")

            opt_results.append({
                "Base Polymer SMILES": base_smi,
                "Tg": round(base_tg, 1),
                "k": round(base_eps, 3),
                "SAScore": round(base_sa, 2),
                "Frontier Size": frontier_size,
                "Avg Tg change (%)": round(avg_tg, 2),
                "Avg k change (%)": round(avg_eps, 2),
                "Avg SAScore": round(avg_sa, 2),
                "Avg Delta SAScore": round(delta_avg_sa, 2),
            })

        plotgm = PlotGenMols(base_eps, base_tg, dir_folder)
        plotgm.plotKDEbyGenerations('tg')
        plotgm.plotKDEbyGenerations('eps')
        plotgm.plotScatterbyGenerations()
        plotgm.plotScatterWithKDEbyGenerations(id=id, version=version)
        plotgm.visualize_gen_bests(mols_per_row=4)
        if 'NS' in version:
            plotgm.visualize_pareto_frontier(mols_per_row=5)

    opt_results_df = pd.DataFrame(opt_results)
    os.makedirs(main_dir, exist_ok=True)
    opt_results_df.to_csv(f'{main_dir}/opt_results.csv', index=False)
    base_polymers_df = pd.DataFrame(base_polymers)
    base_polymers_df.to_csv(f'{main_dir}/base_polymers.csv', index=False)
    visualize_base_polymers(main_dir, base_polymers, mols_per_row=4)
