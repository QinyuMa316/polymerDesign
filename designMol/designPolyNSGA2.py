import os
from pathlib import Path
import random
from typing import Tuple, List, Dict, Any
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
import pandas as pd
from rdkit import Chem
from rdkit.Chem.Draw import rdMolDraw2D
from rdkit.Chem.rdchem import KekulizeException
import torch
from MolOpt.molDataLoader import MolDataset
from torch_geometric.data import DataLoader
from MolOpt.GATmodel import GATModel
from MolOpt.plotGenMols import PlotGenMols
from MolOpt.calSAScore import calSAScore

# ========= 0. Tg 预测模型 =========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = GATModel(num_node_features=7, num_edge_features=6,
                 hidden_dim=128, num_heads=4,
                 dropout=0.2, num_layers=3)

def predict_property(smiles: str, target="Tg") -> float:
    suffix = "_cl_50epo_aug12"
    mdl_path = f"../models_polymer/best_model_{target}{suffix}.pth"
    chkpt = torch.load(mdl_path, map_location=device)
    model.load_state_dict(chkpt["model_state_dict"])
    model.to(device).eval()

    ds  = MolDataset([smiles], [0])
    dl  = DataLoader(ds, batch_size=1, shuffle=False)
    with torch.no_grad():
        for batch in dl:
            out, _ = model(batch.to(device))
            return float(out.cpu().numpy()[0])

# ========= NSGA-II 工具函数 =========
def fast_non_dominated_sort(objs: List[Tuple[float, ...]]) -> List[List[int]]:
    """
    objs: 每个个体的目标向量（全部按“最小化”定义）
    返回：fronts（若干层 Pareto 前沿的索引列表）
    具体：
    1、对每个点 p：
        找出它支配的所有点 S[p]
        统计有多少点支配 p（n[p]）
    2、所有 n[p] = 0 的点 → 第一前沿 Front 0
    3、把 Front 0 中的点“移走”，减少它们支配的点的 n[q]
        原本只被 Front 0 支配的点，n 变成 0 → 成为 第二前沿 Front 1
    4、然后再对 Front 1 做同样的事情，得到 Front 2
    5、直到没有新前沿
    """

    S = [set() for _ in objs]  # 被 i 支配的集合
    n = [0 for _ in objs]      # 支配 i 的个体数量
    fronts = [[]]

    def dominates(a: Tuple[float, ...], b: Tuple[float, ...]) -> bool:
        # a 支配 b：a 至少在一个目标上更好，且所有目标不差
        not_worse = all(x <= y for x, y in zip(a, b))
        strictly_better = any(x < y for x, y in zip(a, b))
        # x支配y：所有维度不差 (x <= y) & 至少一维更好 (x < y)
        return not_worse and strictly_better

    # （1）构造第 1 front

    # 外层p=0: objs[0] = (2,4)
    # 内层q=0: objs[0] 跳过自己continue
    # 内层q=1: objs[1] = (3,3)

    # 判断 x 是否被 y 支配 -> 是否属于 front 0
    for p in range(len(objs)):      # x
        for q in range(len(objs)):  # y
            if p == q:
                continue
            # p=0, q=1, (2,4) vs (3,3)
            # x 支配 y
            if dominates(objs[p], objs[q]):
                S[p].add(q) # y 加入被 x 支配集合
            # x 被 y 支配
            elif dominates(objs[q], objs[p]):
                n[p] += 1   # x 被支配 个数 +1 (y)
            # 对于 p=0, q=1, (2,4) vs (3,3) 两个 if 都不进（互不支配）
        # 如果当前 x 不被任何个体支配，则属于，front 0
        if n[p] == 0:
            fronts[0].append(p)

    # 举例，
    # 0: (2,4)
    # 1: (3,3)
    # 2: (4,2)
    # 3: (5,5)
    #
    # S = [
    #   {3},   # 0 支配 3
    #   {3},   # 1 支配 3
    #   {3},   # 2 支配 3
    #   set()  # 3 不支配任何人
    # ]
    #
    # n = [
    #   0,     # 0 不被任何人支配
    #   0,     # 1 不被任何人支配
    #   0,     # 2 不被任何人支配
    #   3      # 3 被 0,1,2 支配
    # ]
    # fronts = [[0, 1, 2]]
    # # 说明第一前沿 Front0 = [0,1,2]


    # （2）构造后续 front
    i = 0
    while len(fronts[i]) > 0:
        Q = []
        # 第一轮 while：i = 0, fronts[0] = [0,1,2]
        # p 依次 = 0,1,2
        for p in fronts[i]:
            for q in S[p]:
                n[q] -= 1
                if n[q] == 0:
                    Q.append(q)
        i += 1
        fronts.append(Q)
    fronts.pop()  # 最后一层为空，去掉
    return fronts
    # fronts = [[0,1,2], [3]]

def crowding_distance(objs: List[Tuple[float, ...]], front: List[int]) -> Dict[int, float]:
    """
    计算某一前沿层中个体的拥挤距离。
    objs: 所有个体的目标向量（最小化）
    front: 一层前沿的个体索引
    返回：该层内每个索引的拥挤距离
    """
    if not front:
        return {}
    m = len(objs[0])  # 目标数
    dist = {i: 0.0 for i in front}
    # m：目标维度数，比如 (EPS, -Tg) 就是 2 维。
    # dist：存每个个体的拥挤距离，初始全是 0。

    for k in range(m):
        # 以第 k 目标对该层排序
        front_sorted = sorted(front, key=lambda idx: objs[idx][k])
        fmin = objs[front_sorted[0]][k]
        fmax = objs[front_sorted[-1]][k]
        # 边界解给无穷大
        dist[front_sorted[0]]  = float('inf')
        dist[front_sorted[-1]] = float('inf')
        if fmax == fmin:
            continue
        for j in range(1, len(front_sorted) - 1):
            prev_obj = objs[front_sorted[j - 1]][k]
            next_obj = objs[front_sorted[j + 1]][k]
            dist[front_sorted[j]] += (next_obj - prev_obj) / (fmax - fmin)
    return dist

def tournament_select(pool: List[int],
                              ranks: Dict[int, int],
                              cdists: Dict[int, float],
                              objs: List[Tuple[float, ...]],
                              aux_list: List[Dict[str, Any]],
                              gen: int,
                              t_id: int,
                              log_fp=None,
                              verbose: bool=False
                              ) -> int:
    """
    二元锦标赛（带详细输出/写日志）
    """
    # 1) 从 parent_pool 随机抽两个不同候选
    a, b = random.sample(pool, 2)
    # 2) 看 rank
    ra, rb = ranks[a], ranks[b]
    da, db = cdists.get(a, 0.0), cdists.get(b, 0.0)

    # 决定胜者
    if ra < rb:
        winner = a
        reason = "better rank"
    elif rb < ra:
        winner = b
        reason = "better rank"
    else:
        if da > db:
            winner = a
            reason = "same rank, higher crowding"
        elif db > da:
            winner = b
            reason = "same rank, higher crowding"
        else:
            winner = a if random.random() < 0.5 else b
            reason = "tie, random"

    # 3) 如果 verbose=True，就 print 出来
    # if verbose:
    #     print(f"\n[Gen {gen}] Tournament #{t_id}:")
    #     # print(f"  candA idx={a} rank={ra} crowd={da:.3f} obj={objs[a]} eps={aux_list[a]['eps']:.2f} tg={aux_list[a]['tg']:.2f}")
    #     # print(f"  candB idx={b} rank={rb} crowd={db:.3f} obj={objs[b]} eps={aux_list[b]['eps']:.2f} tg={aux_list[b]['tg']:.2f}")
    #
    #     print(f"  candA idx={a} rank={ra} crowd={da:.3f} eps={aux_list[a]['eps']:.2f} tg={aux_list[a]['tg']:.2f}")
    #     print(f"  candB idx={b} rank={rb} crowd={db:.3f} eps={aux_list[b]['eps']:.2f} tg={aux_list[b]['tg']:.2f}")
    #     print(f"  -> winner idx={winner} ({reason})")
    #
    # # 4) 写入 parent_log_fp，记下本次锦标赛
    # if log_fp is not None:
    #     rec = {
    #         "generation": gen,
    #         "tournament_id": t_id,
    #         "candA": {
    #             "idx": a, "rank": ra, "crowding": da,
    #             "obj": objs[a], "tg": aux_list[a]["tg"], "eps": aux_list[a]["eps"],
    #             "smiles": aux_list[a]["smiles"]
    #         },
    #         "candB": {
    #             "idx": b, "rank": rb, "crowding": db,
    #             "obj": objs[b], "tg": aux_list[b]["tg"], "eps": aux_list[b]["eps"],
    #             "smiles": aux_list[b]["smiles"]
    #         },
    #         "winner": {"idx": winner, "reason": reason}
    #     }
    #     json.dump(rec, log_fp, ensure_ascii=False)
    #     log_fp.write("\n")

    return winner


class NSGA2():
    def __init__(self, base_smi,
                 base_tg, base_eps,
                 frag_smiles, # target,
                 dir_folder,
                 w_tg=1.0, w_eps=1.0,
                 N_GEN = 50, POP_SIZE = 40,
                 KEEP_TOP_K = 10, MUT_RATE = 0.2,
                 NO_REPLACE_ID = -1,
                 ):
        """
        NSGA-II 参数与原 GA 一致，含义相同。
        """
        self.base_smi = base_smi
        self.base_mol = Chem.MolFromSmiles(self.base_smi)
        self.base_tg  = base_tg
        self.base_eps = base_eps

        frag_smi, frag_mols = self.check_fp_validity(frag_smiles)
        self.frag_mols   = frag_mols
        self.frag_smiles = frag_smi

        self.replace_sites = self.get_replace_sites()
        self.FRAG_RANGE = range(len(frag_smi))
        self.GENE_LEN = len(self.replace_sites)

        self.N_GEN       = N_GEN
        self.POP_SIZE    = POP_SIZE
        self.KEEP_TOP_K  = KEEP_TOP_K  # 在 NSGA-II 中主要用于“父本池”多样性保留（可不用，但保留参数）
        self.MUT_RATE    = MUT_RATE
        self.NO_REPLACE_ID = NO_REPLACE_ID
        self.target      = 'multi'
        self.dir_folder  = dir_folder
        os.makedirs(self.dir_folder, exist_ok=True)

        self.w_tg  = w_tg
        self.w_eps = w_eps

        # 标准化区间（用于“可视化用的标量分数”）
        # if target == 'multi':
        self.eps_max = 4.0
        self.eps_min = 2.0
        self.tg_max  = 800.0
        self.tg_min  = 400.0

        # 非法个体的目标向量（最小化问题）
        # if target == 'multi':
        #     self.bad_obj = (float('inf'), float('inf'))
        # else:
        #     self.bad_obj = (float('inf'),)
        self.bad_obj = (float('inf'), float('inf'))

    # ========= RDKit & 结构处理 =========
    def get_replace_sites(self):
        star_idx = [at.GetIdx() for at in self.base_mol.GetAtoms()
                    if at.GetAtomicNum() == 0]
        replace_sites = []
        for at in self.base_mol.GetAtoms():
            if at.GetAtomicNum() == 6 and at.GetIsAromatic() \
                    and at.GetIdx() not in star_idx and at.GetTotalNumHs() > 0:
                replace_sites.append(at.GetIdx())
        return replace_sites

    def check_fp_validity(self, frag_smiles):
        frag_smi, frag_mols = [], []
        for s in frag_smiles:
            m = Chem.MolFromSmiles(s)
            if m is None:
                print(f"[WARN] 片段 {s} 解析失败，已忽略")
                continue
            try:
                Chem.SanitizeMol(m)
                frag_smi.append(s)
                frag_mols.append(m)
            except Exception:
                print(f"[WARN] 片段 {s} 消毒失败，已忽略")
                continue
        return frag_smi, frag_mols

    def attach_fragment(self, base: Chem.Mol, atom_idx: int, frag: Chem.Mol) -> Chem.Mol | None:
        if frag is None or base is None:
            return None
        combo  = Chem.CombineMols(base, frag)
        emol   = Chem.EditableMol(combo)
        offset = base.GetNumAtoms()
        emol.AddBond(atom_idx, offset, Chem.BondType.SINGLE)
        try:
            new = emol.GetMol()
            Chem.SanitizeMol(new)
            return new
        except (KekulizeException, ValueError):
            return None

    def genome_to_smiles(self, genome: Tuple[int]) -> str | None:
        mol = self.base_mol
        for site_idx, frag_idx in enumerate(genome):
            if frag_idx == -1:
                continue
            mol = self.attach_fragment(
                mol, self.replace_sites[site_idx], self.frag_mols[frag_idx]
            )
            if mol is None:
                return None
        # 确保仍有两个 *
        if sum(1 for at in mol.GetAtoms() if at.GetAtomicNum() == 0) != 2:
            return None
        return Chem.MolToSmiles(mol, isomericSmiles=False)

    def random_genome(self) -> Tuple[int]:
        return tuple(random.choice(self.FRAG_RANGE) if random.random() < .3 else self.NO_REPLACE_ID
                     for _ in range(self.GENE_LEN))

    def crossover(self, p1: Tuple[int], p2: Tuple[int]) -> Tuple[int]:
        cut = random.randint(1, self.GENE_LEN-1) if self.GENE_LEN > 1 else 1
        return p1[:cut] + p2[cut:]

    def mutate(self, g: Tuple[int]) -> Tuple[int]:
        g = list(g)
        for i in range(self.GENE_LEN):
            if random.random() < self.MUT_RATE:
                g[i] = random.choice([self.NO_REPLACE_ID, *self.FRAG_RANGE])
        return tuple(g)

    # ========= 目标与可视化分数 =========
    def _clip_mm(self, x, xmin, xmax):
        if xmax == xmin:
            return 0.0
        return max(0.0, min(1.0, (x - xmin) / (xmax - xmin)))

    def objectives(self, genome: Tuple[int]) -> Tuple[Tuple[float, ...], Dict[str, Any]]:
        """
        返回：(obj_vector, aux_info)
        obj_vector: 按“最小化”方向的目标元组
        aux_info: 附加信息（用于记录/可视化），包括 smi、tg、eps、desirability 等
        """
        smi = self.genome_to_smiles(genome)
        if smi is None:
            # if self.target == 'multi':
            #     aux = {"smiles": None, "tg": None, "eps": None, "des": 0.0}
            # else:
            #     aux = {"smiles": None, "tg": None, "eps": None, "des": 0.0}
            aux = {"smiles": None, "tg": None, "eps": None, "des": 0.0}
            return self.bad_obj, aux

        try:
            # if self.target == 'multi':
            tg  = predict_property(smi, "Tg")
            eps = predict_property(smi, "EPS")
            # NSGA-II 目标：min (EPS, -Tg)
            obj = (eps, -tg)

            # 可视化用标量分数（与原 fitness 一致）
            z_t   = self._clip_mm(tg,  self.tg_min,  self.tg_max)
            z_eps = self._clip_mm(eps, self.eps_min, self.eps_max)
            d_t   = z_t
            d_eps = 1.0 - z_eps
            w_sum = self.w_tg + self.w_eps
            desirability = (d_t**self.w_tg * d_eps**self.w_eps) ** (1.0 / w_sum)
            aux = {"smiles": smi, "tg": tg, "eps": eps, "des": desirability}
            return obj, aux
            # ！NSGA-II 目标：min (EPS, -Tg)
            # obj = (eps, -tg)
            # aux = {"smiles": smi, "tg": tg, "eps": eps, "des": desirability}
            # desirability 即 naive GA 计算的 fitness

            # elif self.target == 'Tg':
            #     tg  = predict_property(smi, "Tg")
            #     obj = (-tg,)  # 最大化 Tg -> 最小化 -Tg
            #     aux = {"smiles": smi, "tg": tg, "eps": None, "des": tg}  # 可视化直接用 Tg
            #     return obj, aux
            #
            # else:  # self.target == 'EPS'
            #     eps = predict_property(smi, "EPS")
            #     obj = (eps,)
            #     aux = {"smiles": smi, "tg": None, "eps": eps, "des": -eps}  # 可视化用 -EPS 越大越好
            #     return obj, aux

        except Exception:
            return self.bad_obj, {"smiles": None, "tg": None, "eps": None, "des": 0.0}

    # ========= 绘图 =========
    def plot_ga_process(self, best_hist, avg_hist, pop_fitness_hist=None):
        # 这里 best_hist/avg_hist 实际是“desirability”历史，而不是 NSGA-II 的 rank
        dark_blue = '#7A9EBA'
        dark_red = '#D28C8C'
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 12,
            'figure.titlesize': 12
        })
        plt.figure(figsize=(self.N_GEN * 7/50, 5))

        if pop_fitness_hist:
            for g_idx, gen_scores in enumerate(pop_fitness_hist):
                plt.scatter([g_idx] * len(gen_scores),
                            gen_scores,
                            marker='o', facecolors='none',
                            edgecolors='gray', alpha=0.5, s=25,
                            label="Population" if g_idx == 0 else "")

        plt.plot(best_hist, label="Best Score", linewidth=2, color=dark_red)
        plt.plot(avg_hist, label="Avg Score", linewidth=2, color=dark_blue)
        plt.xlim(0 - 1, self.N_GEN - 1 + 1)
        plt.ylim(0, 1 if self.target=='multi' else None)
        plt.xlabel("Generations", labelpad=5)
        plt.ylabel("Fitness", labelpad=5)
        plt.title("NSGA-II Search Process", size=12) #  (scalarized for visualization)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dir_folder}/ga_progress.png", dpi=600)
        plt.close()

    def smi_to_png(self, smi, legend, png_name,
                           img_size=(700, 400),
                           legend_font_size=24):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")
        drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
        opts = drawer.drawOptions()
        opts.legendFontSize = legend_font_size
        drawer.DrawMolecule(mol, legend=legend or "")
        drawer.FinishDrawing()
        out_png = Path(png_name)
        out_png.write_bytes(drawer.GetDrawingText())

    # ———— v5 修改
    def plot_pareto_evolution(self, all_aux_by_gen, all_fronts_by_gen):
        """
        绘制所有代的整体散点 & 每代 Pareto Front 0 演化轨迹（颜色随代数变化）
        """

        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 12,
            'figure.titlesize': 12
        })

        out_png = f"{self.dir_folder}/pareto_frontier_evolution.png"

        num_gens = len(all_aux_by_gen)
        gens = list(range(num_gens))

        # 颜色映射：generation → 颜色
        norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))
        cmap = mpl.colormaps['viridis']  # 与用户要求保持一致
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])

        fig, ax = plt.subplots(figsize=(7, 5))

        # -----------------------------------------------------------
        # ① 画所有代所有点（灰色淡化背景）
        # -----------------------------------------------------------
        for aux_list in all_aux_by_gen:
            xs, ys = [], []
            for aux in aux_list:
                if aux["smiles"] is None:
                    continue
                xs.append(aux["eps"])
                ys.append(aux["tg"])
            ax.scatter(xs, ys, alpha=0.5, color='#999999', s=20)

        # -----------------------------------------------------------
        # ② 随代数变化颜色：画各代 Front 0 曲线
        # -----------------------------------------------------------
        for gen_idx, (aux_list, fronts) in enumerate(zip(all_aux_by_gen, all_fronts_by_gen)):
            if not fronts:
                continue

            front0 = fronts[0]
            fxs, fys = [], []
            for idx in front0:
                aux = aux_list[idx]
                if aux["smiles"] is None:
                    continue
                fxs.append(aux["eps"])
                fys.append(aux["tg"])

            if not fxs:
                continue

            # 按 EPS 从小到大排序，让轨迹连线更自然
            front_sorted = sorted(zip(fxs, fys), key=lambda t: t[0])
            fxs, fys = zip(*front_sorted)

            # 前沿颜色依据 generation 映射
            color = cmap(norm(gen_idx))
            ax.plot(
                fxs, fys,
                color=color,
                linewidth=1.6,
                alpha=0.85
            )

        # -----------------------------------------------------------
        # ③ 坐标轴 / 标题 / colorbar
        # -----------------------------------------------------------
        ax.set_xlabel("k (minimize ←)")
        ax.set_ylabel("Tg (maximize →)")
        ax.set_title("Pareto Frontier Evolution Across Generations")
        # ax.grid(True) # 网格关闭

        # Base Molecular 点必须先画！
        ax.scatter(self.base_eps, self.base_tg,
                   color='red', marker='*', s=100,
                   label='Base Molecular')

        # colorbar：对应 generation 变化
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)
        cbar.set_label("Generation")

        fig.tight_layout()
        plt.legend(loc='lower left')  # 放在右下角 —— upper/lower right/left
        fig.savefig(out_png, dpi=600)
        plt.close(fig)

    # ========= NSGA-II 主循环 =========
    def run_ga(self, plot=True,
               # debug_select=False,
               # save_parent_viz=False
               ):
        """
        NSGA-II 主循环（带父母选择过程可视化/日志）

        参数：
          plot: 是否画“标量分数（des）进化曲线”
          debug_select: 是否在控制台逐次打印 tournament 选择细节
          save_parent_viz: 是否每代输出 Pareto 前沿 + rank/crowding + 父母选择箭头图
                            （建议只在 target='multi' 时打开）

        返回：
          - target == 'multi' : (best_smi, (best_eps, best_tg))
          - target == 'Tg'    : (best_smi, best_tg)
          - target == 'EPS'   : (best_smi, best_eps)
        """

        # --- 历史记录（用于绘图的“标量分数 des”） ---
        best_hist, avg_hist = [], []
        pop_score_hist = []

        # ———— v4 修改
        all_aux_by_gen: List[List[Dict[str, Any]]] = []
        all_fronts_by_gen: List[List[List[int]]] = []
        # ———— v4 修改

        # --- 打开 jsonl 文件 ---
        jsonl_all = f"{self.dir_folder}/ga_population.jsonl"  # 全部“合法”个体
        jsonl_best = f"{self.dir_folder}/ga_best_per_gen.jsonl"  # 每代代表最佳（按 des 最大）
        jf_all = open(jsonl_all, "w", encoding="utf-8")
        jf_best = open(jsonl_best, "w", encoding="utf-8")

        # --- 父母选择日志 ---
        # parent_log_path = f"{self.dir_folder}/nsga2_parent_selection.jsonl"
        # parent_log_fp = open(parent_log_path, "w", encoding="utf-8")

        # 初始种群
        pop = [self.random_genome() for _ in range(self.POP_SIZE)]

        # 记录代表性最优（用于最终输出）
        best_rep_score = -1e18
        # best_rep_smi = None
        # best_rep_eps = None
        # best_rep_tg = None

        for gen in range(self.N_GEN):

            # ========= 1) 评估当前代（得到 objs & aux_list） =========
            objs, aux_list = [], []
            for g in pop:
                obj, aux = self.objectives(g)
                # obj = (eps, -tg)
                # aux = {"smiles": smi, "tg": tg, "eps": eps, "des": desirability}
                objs.append(obj)
                aux_list.append(aux)

            # ========= 2) 快速非支配排序 =========
            fronts = fast_non_dominated_sort(objs)
            # fronts = [[0,1,2], [3]]

            # ———— v4 修改
            all_aux_by_gen.append(aux_list)
            all_fronts_by_gen.append(fronts)
            # ———— v4 修改

            # ========= 3) 计算 rank / crowding =========
            ranks = {}
            cdists = {}
            for i, fr in enumerate(fronts):
                for idx in fr:
                    ranks[idx] = i
                dist = crowding_distance(objs, fr)
                cdists.update(dist)

            # ========= 4) 记录本代所有合法个体 =========
            gen_scores = []
            for idx, (g, obj, aux) in enumerate(zip(pop, objs, aux_list)):
                if aux["smiles"] is not None:
                    des = aux["des"] if aux["des"] is not None else 0.0
                    gen_scores.append(des)
                    record = {
                        "smiles": aux["smiles"],
                        "generation": gen,
                        "rank": ranks.get(idx, 1e9),
                        "crowding": cdists.get(idx, 0.0),
                        "tg": aux["tg"],
                        "eps": aux["eps"],
                        "score": des
                    }
                    json.dump(record, jf_all, ensure_ascii=False)
                    jf_all.write("\n")

            # ========= 5) 每代“代表性最佳”（按 des 最大） =========
            if gen_scores:
                best_des = -1e18
                best_idx = None
                for idx, aux in enumerate(aux_list):
                    if aux["smiles"] is None:
                        continue
                    des = aux["des"] if aux["des"] is not None else -1e18
                    if des > best_des:
                        best_des = des
                        best_idx = idx

                if best_idx is not None:
                    auxb = aux_list[best_idx]
                    json.dump({
                        "smiles": auxb["smiles"],
                        "generation": gen,
                        "tg": auxb["tg"],
                        "eps": auxb["eps"],
                        "score": auxb["des"]
                    }, jf_best, ensure_ascii=False)
                    jf_best.write("\n")

                    # 更新全局代表最优
                    if auxb["des"] is not None and auxb["des"] > best_rep_score:
                        best_rep_score = auxb["des"]
                        # best_rep_smi = auxb["smiles"]
                        # best_rep_eps = auxb["eps"]
                        # best_rep_tg = auxb["tg"]

                # 绘图历史
                best_hist.append(max(gen_scores))
                avg_hist.append(sum(gen_scores) / len(gen_scores))
                pop_score_hist.append(gen_scores.copy())
            else:
                best_hist.append(0.0)
                avg_hist.append(0.0)
                pop_score_hist.append([])

            # ========= 6) 产生子代：基于(rank, crowding)锦标赛选父母 =========

            # 1、构造索引列表 + 按 (rank, -crowding) 排序
            # all_indices = [0, 1, 2, ..., N-1]
            # ranks[i]：个体 i 属于第几前沿（0 是最优，1 次之…）
            # cdists[i]：个体 i 在自己那一层的拥挤距离（越大越“稀疏”）
            all_indices = list(range(len(pop)))
            sorted_pool = sorted(
                all_indices,
                key=lambda i: (ranks.get(i, 1e9), -cdists.get(i, 0.0))
                # 先按 rank 升序，再按 crowding 降序
            )
            parent_pool = sorted_pool[:max(self.POP_SIZE, 2)]
            # 只取排序前 POP_SIZE 个作为“父母候选池”
            # 2、锦标赛选择 + 交叉变异生成子代
            children = []
            parent_pairs = []  # 记录每次交叉用到的父母索引（用于可视化）
            t_id = 0  # tournament 计数
            # 只要 children 数量还没达到 POP_SIZE，就继续造孩子
            # 每次造一个孩子，需要两次锦标赛：
            #     第1次：选出父亲 p1_idx
            #     第2次：选出母亲 p2_idx
            while len(children) < self.POP_SIZE:
                t_id += 1
                p1_idx = tournament_select(
                    parent_pool, ranks, cdists,
                    objs, aux_list,
                    gen=gen, t_id=t_id,
                    # log_fp=parent_log_fp,
                    # verbose=debug_select
                )

                t_id += 1
                p2_idx = tournament_select(
                    parent_pool, ranks, cdists,
                    objs, aux_list,
                    gen=gen, t_id=t_id,
                    # log_fp=parent_log_fp,
                    # verbose=debug_select
                )
                # todo: 把 (p1_idx, p2_idx) 记录下来 → 后面画图用
                parent_pairs.append((p1_idx, p2_idx))

                c = self.mutate(self.crossover(pop[p1_idx], pop[p2_idx]))
                children.append(c)

            # 每代父母选择可视化（只建议 multi 时开）
            # if save_parent_viz and self.target == "multi":
            #     self.plot_front_and_parents(aux_list, ranks, cdists, parent_pairs, gen)

            # ========= 7) 环境选择：父+子=2N，重新做NSGA-II选择N个 =========
            combined = pop + children
            # pop：这一代的父代个体（N 个）
            # children：刚刚用交叉+变异生成的子代（N 个）
            # 现在 combined 里有 2 * POP_SIZE 个体。
            # NSGA-II 的策略是：
            # 不直接丢弃父代，而是把父+子一起拿出来用 Pareto 排序，
            # 重新选出最好的 N 个 → “精英策略（elitism）”。
            comb_objs, comb_aux = [], []
            for g in combined:
                o, a = self.objectives(g)
                # o, obj：目标向量，比如 multi 时是 (eps, -tg)（全部按最小化）
                # a, aux：辅助信息，包括：smiles, tg, eps, des
                comb_objs.append(o)
                comb_aux.append(a)
                # comb_objs[i]：combined 中第 i 个个体的 obj
                # comb_aux[i]：对应的属性
            comb_fronts = fast_non_dominated_sort(comb_objs)
            # comb_fronts[0] → 最优的那一层（Front 0）
            # comb_fronts[1] → 次优
            # 用“层级 + 拥挤度”来决定谁能留下（核心）
            new_pop = []
            for fr in comb_fronts:
                # fr 是一个索引列表，比如 fr = [0,4,5,...]，表示这些个体在同一前沿层。
                if len(new_pop) + len(fr) <= self.POP_SIZE:
                    # 如果这一层“全部装得下”，就全收
                    new_pop.extend([combined[i] for i in fr])
                else:
                    # 如果这一层“装不下了”，就用 crowding 截断
                    dist = crowding_distance(comb_objs, fr)
                    fr_sorted = sorted(fr, key=lambda i: dist.get(i, 0.0), reverse=True)
                    remain = self.POP_SIZE - len(new_pop)
                    new_pop.extend([combined[i] for i in fr_sorted[:remain]])
                    break
            pop = new_pop  # 下一代

        # ========= 关闭文件 =========
        jf_all.close()
        jf_best.close()
        # parent_log_fp.close()

        # ========= 画标量分数曲线（des进化，仅用于观察趋势） =========
        if plot:
            self.plot_ga_process(best_hist, avg_hist, pop_score_hist)
            # ———— v4 修改
            self.plot_pareto_evolution(all_aux_by_gen, all_fronts_by_gen)
            # ———— v4 修改

        # ———— hyperparam_v2_1_22 修改
        # ========= 在最终种群上重新计算 Pareto Frontier =========
        final_objs, final_aux = [], []
        for g in pop:
            o, a = self.objectives(g)
            final_objs.append(o)
            final_aux.append(a)

        final_fronts = fast_non_dominated_sort(final_objs)
        front0_idx = final_fronts[0] if final_fronts else []

        # ========= 收集 Pareto frontier 结果 =========
        pareto_solutions = []
        for idx in front0_idx:
            aux = final_aux[idx]
            if aux["smiles"] is None:
                continue
            pareto_solutions.append({
                "smiles": aux["smiles"],
                "tg": aux["tg"],
                "eps": aux["eps"],
                "obj": final_objs[idx],  # (eps, -tg)
                "rank": 0
            })

        # ========= 保存 Pareto frontier 到 jsonl =========
        pareto_path = f"{self.dir_folder}/pareto_frontier.jsonl"
        with open(pareto_path, "w", encoding="utf-8") as fp:
            for rec in pareto_solutions:
                json.dump(rec, fp, ensure_ascii=False)
                fp.write("\n")

        # ========= 画最终 Pareto frontier 图 =========
        # if self.target == "multi":
        #     self.plot_pareto_frontier(final_aux, final_fronts,
        #                               gen=self.N_GEN - 1, tag="final")

        # ========= 仍可输出 base polymer 的结构图（参考用） =========
        self.smi_to_png(
            smi=self.base_smi,
            legend=f"Tg: {self.base_tg:.1f} K, k: {self.base_eps:.3f}",
            png_name=f"{self.dir_folder}/base_polymer.png"
        )

        # ========= 不再输出单个 des 最高的 best_polymer 结构图 =========
        # （因为 NSGA-II 不应只返回一个解）

        # ========= 返回 Pareto frontier 列表 =========
        return pareto_solutions
        # ———— hyperparam_v2_1_22 修改


if __name__ == "__main__":

    from designPolyGA_main import frag_smiles
    RUN_FROM_SCRATCH = True

    N_GEN = 50
    POP_SIZE = 50
    # target = 'multi'
    dir_folder = f'NSGA2_{N_GEN}GEN_{POP_SIZE}POP'

    # ========= 1. 核心骨架 & 替换位点 =========
    base_smi = "*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1"
    # base_mol = Chem.MolFromSmiles(base_smi)
    random.seed(2025)
    base_tg = predict_property(base_smi, 'Tg')
    base_eps = predict_property(base_smi, 'EPS')
    base_sa = calSAScore(base_smi)
    print(f"Base polymer Tg: {base_tg:.1f} K, k: {base_eps:.3f}, SAScore: {base_sa:.2f}")
    GA = NSGA2(base_smi, base_tg, base_eps,
               frag_smiles, # target,
               dir_folder, N_GEN, POP_SIZE)
    # best_smi, best_prop = GA.run_ga()

    if RUN_FROM_SCRATCH:
        print('Running NSGA-II ...')
        # 如果你想看“rank+crowding 如何选父母”的全过程，可开启下面两个开关
        # ———— hyperparam_v2_1_22 修改
        pareto_solutions = GA.run_ga(
            plot=True,
            # debug_select=True,      # True 会逐次打印 tournament 选择细节（输出很多）
            # save_parent_viz=True     # True 会每代输出 Pareto前沿+父母选择图
        )
        print("=== NSGA-II Pareto Frontier ===")
        print(f"Frontier size = {len(pareto_solutions)}")
        # pareto_solutions
        # "smiles": aux["smiles"],
        # "tg": aux["tg"],
        # "eps": aux["eps"],
        # "obj": final_objs[idx],  # (eps, -tg)
        # "rank": 0

        delta_tg_list = []
        delta_eps_list = []
        sa_list = []
        # print(f'Show top-5 mols:')
        # for i, sol in enumerate(pareto_solutions[:5]):  # 只打印前5个
        for i, sol in enumerate(pareto_solutions):  # 只打印前5个
            cur_tg = sol["tg"]
            cur_eps = sol["eps"]
            cur_smi = sol["smiles"]
            cur_sa = calSAScore(cur_smi)
            delta_tg = (cur_tg - base_tg) / base_tg * 100
            delta_eps = (cur_eps - base_eps) / base_eps * 100
            delta_tg_list.append(delta_tg)
            delta_eps_list.append(delta_eps)
            sa_list.append(cur_sa)

            print(f"ID: {i} Tg: {cur_tg:.1f} K, k: {cur_eps:.2f}, SAScore: {cur_sa:.2f}")
            print(f"SMILES: {sol['smiles']}")
            print(f'Improvement ratio: Tg: {delta_tg:.2f}%, k: {delta_eps:.2f}%')

        avg_tg = sum(delta_tg_list) / len(delta_tg_list)
        avg_eps = sum(delta_eps_list) / len(delta_eps_list)
        avg_sa = sum(sa_list) / len(sa_list)

        # print("=== Average Improvement ===")
        print(f"Avg Tg improvement: {avg_tg:.2f}%")
        print(f"Avg k decrease: {avg_eps:.2f}%")
        print(f"Avg SAScore: {avg_sa:.2f}")

        # ———— hyperparam_v2_1_22 修改

    plotgm = PlotGenMols(base_eps, base_tg, dir_folder)
    plotgm.plotKDEbyGenerations('tg')
    plotgm.plotKDEbyGenerations('eps')
    plotgm.plotScatterbyGenerations()
    plotgm.visualize_gen_bests(mols_per_row=5)
    plotgm.visualize_pareto_frontier(mols_per_row=5)

    # NSGA2v2
    # ———— v2:
    # 每一代输出 Pareto rank 和 crowding distance
    # 打印每次锦标赛（tournament）比较的两名选手 + 胜者
    # 把“父母选择过程”写成 jsonl 日志
    # 画图：在 (EPS, Tg) 空间里显示每代的前沿 + crowding，并用箭头标出被选作父代的点
    # todo: 图片nsga2_front_gen包含了哪些信息，给我解读一下
    # todo: def plot_front_and_parents
    # ———— hyperparam_v2_1_22:
    # NSGA-II 的本意不是给出一个“综合分数最高”的解，而是给出一整条 Pareto 前沿上的权衡解集合。
    # 我们现在把代码改成：
    # 最终返回 Front 0（Pareto frontier）上的一系列分子
    # 保存 Pareto frontier 的 jsonl
    # 生成一张显示 Pareto frontier 的图（而不是单个 des 最高分子结构图）
    # （可选）仍保留 base 分子的结构图输出，但不再输出“best_polymer”单体图。
    # https://www.youtube.com/watch?v=SL-u_7hIqjA
    # ———— v4:
    # v3只画了最终一代（final population）的 Front 0，没有把所有 generation 的所有种群/前沿演化都保存进去。
    # 方案B：保存一张“所有代的所有点 + 每代前沿轨迹”总览图 def plot_pareto_evolution
    # ———— v5:
    # 进一步修改：def plot_pareto_evolution
    # 使用 colormap（viridis） 随着 generation 改变颜色
    # 每一代的 Front 0 曲线颜色渐变
    # 在图右侧显示 colorbar（表示代数）
    # 删除：—— def plot_pareto_frontier
    # ———— v6:
    # 可视化所有pareto_solutions & 评估pareto_solutions每个mol的SA score (在 visualize_pareto_frontier 中计算，并按照升序排列)

    # Base polymer Tg: 559.6 K, k: 3.159, SAScore: 3.29
    # ———— 50 GEN 40 POP
    # Frontier size = 33
    # Avg Tg improvement: 7.19%
    # Avg k decrease: -19.31%
    # Avg SAScore: 4.68
    # ———— 50 GEN 50 POP 和 40 POP没区别

