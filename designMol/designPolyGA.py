"""
用 RDKit CalcLabuteASA（近似分子可接触表面积，ASA 越大通常表示体积/空间占用越大）来衡量“空间位阻”。
先计算基础骨架的 ASA，然后允许一定的放大倍数（默认 1.5 倍，可自行调整）。
只要生成分子的 ASA > 阈值，就视为“位阻过大”→ 返回 None，在 GA 中等同于非法个体被淘汰。
"""
from rdkit.Chem import rdMolDescriptors
# 用于计算 Labute ASA

import os
from pathlib import Path
import random
from typing import Tuple
import matplotlib.pyplot as plt
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

class GeneticAlgorithm():
    def __init__(self, base_smi,
                 base_tg, base_eps,
                 frag_smiles, dir_folder, # target='multi',
                 w_tg=1.0, w_eps=1.0,
                 N_GEN = 50, POP_SIZE = 40,
                 KEEP_TOP_K = 10, MUT_RATE = 0.2,
                 NO_REPLACE_ID = -1,
                 ):
        """
        N_GEN = 50  # total interation
        POP_SIZE = 40  # population size for evert generation
        KEEP_TOP_K = 10  # 保留fitness最高的top_k个
        MUT_RATE = 0.2  # mutation比例
        NO_REPLACE_ID = -1
        """
        self.base_smi = base_smi
        self.base_mol = Chem.MolFromSmiles(self.base_smi)
        self.base_tg = base_tg
        self.base_eps = base_eps
        frag_smi, frag_mols = self.check_fg_validity(frag_smiles)
        self.frag_mols = frag_mols
        self.frag_smiles = frag_smi
        self.replace_sites = self.get_replace_sites()
        self.FRAG_RANGE = range(len(frag_smi))  # range(len(frag_smiles))
        # 若全部有效，则 len(frag_smiles) == len(frag_smi) == len(frag_mols)
        # frag_smiles的长度，即可选官能团的总数量
        self.GENE_LEN = len(self.replace_sites)
        self.N_GEN = N_GEN
        self.POP_SIZE = POP_SIZE
        self.KEEP_TOP_K = KEEP_TOP_K
        self.MUT_RATE = MUT_RATE
        self.NO_REPLACE_ID = NO_REPLACE_ID
        self.target = 'multi'
        self.dir_folder = dir_folder
        os.makedirs(self.dir_folder, exist_ok=True)
        # if target == 'multi' or target == 'Tg':
        #     self.maximize = True
        # else:
        #     self.maximize = False
        self.maximize = True
        # self.maximize = maximize
        # “极差分”值：非法个体给一个最差 fitness，使其必被淘汰
        self.bad_fitness = 0.0 if self.maximize else 1e6
        # if target == 'multi':
        #     # data_path = 'data_mod/polyimides_pred_prop_base_str_std.csv'
        #     # df = pd.read_csv(data_path)
        #     # self.eps_max = df['EPS'].max()  # 3.3516688
        #     # self.eps_min = df['EPS'].min()  # 3.0387225
        #     # self.tg_max = df['Tg'].max()  # 716.16406
        #     # self.tg_min = df['Tg'].min()  # 510.6628
        self.eps_max = 4.0
        self.eps_min = 2.0
        self.tg_max = 800.0
        self.tg_min = 400.0

        self.w_tg = w_tg
        self.w_eps = w_eps
        # # ==== Steric-MOD BEGIN ====
        # # 计算基础骨架的可接触表面积 (Labute ASA)，并设置允许的最大值
        # self.base_asa = rdMolDescriptors.CalcLabuteASA(self.base_mol)
        # self.max_asa = self.base_asa * 1.5  # 允许最多放大 50%，可按需求调节
        # # ==== Steric-MOD END   ====

    # ==== Steric-MOD BEGIN ====
    # def _steric_ok(self, mol: Chem.Mol) -> bool:
    #     """返回 True 表示位阻可接受；False 表示过大"""
    #     if mol is None:
    #         return False
    #     asa = rdMolDescriptors.CalcLabuteASA(mol)
    #     return asa <= self.max_asa

    # ==== Steric-MOD END   ====

    def get_replace_sites(self):
        star_idx = [at.GetIdx() for at in self.base_mol.GetAtoms()
                    if at.GetAtomicNum() == 0]
        # * 在 RDKit 里是 原子序号 0 的“哑原子”；找到这两个 * 的索引
        replace_sites = []
        for at in self.base_mol.GetAtoms():
            if at.GetAtomicNum() == 6 and at.GetIsAromatic() and at.GetIdx() not in star_idx and at.GetTotalNumHs() > 0:
                replace_sites.append(at.GetIdx())
        return replace_sites

    # ========= 2. 片段库（过滤无效） =========

    def check_fg_validity(self, frag_smiles):
        # 一串想尝试的官能团 SMILES。每个代表一个可能的突变值。
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
        # RDKit 解析 + SanitizeMol() 做价电子、芳香性校验。不合法就打印警告并丢弃。
        # 过滤后的列表 frag_mols 与 frag_smi 索引一一对应，FRAG_RANGE 后面用于随机采样。
        return frag_smi, frag_mols

    # ========= 3. 分子拼接工具 =========
    def attach_fragment(self, base: Chem.Mol, atom_idx: int, frag: Chem.Mol) -> Chem.Mol | None:
        # 在 base 的 atom_idx 单键连接 frag(0)；若非法返回 None???
        """
        mol = attach_fragment(base = mol,
                             atom_idx = replace_sites[site_idx],
                             frag = frag_mols[frag_idx]
                             )
        把官能团frag_mols[frag_idx]接到分子的replace_sites[site_idx]这个位点上

        把 骨架 base 和 官能团 frag 先 Combine 成一个“并列分子”
        用 EditableMol 在 atom_idx 与 frag 的 0 号原子之间加单键。
        offset = base.GetNumAtoms() 表示 “fragment 部分的原子索引整体偏移”。
        Chem.SanitizeMol() 再次检查：如果价电子或芳香性冲突就抛异常；捕获后返回 None，遗传算法会将其适应度设为极小（-1e6）。
        """
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
        """
        genome 是元组，长度等于可替换位点数；
        元素值 -1 表示“该位点保持原状”；
        其余非负整数是片段库 frag_mols 的索引。
        返回值：
        若成功构建合法分子，就给出对应 SMILES 字符串；
        出错或违反约束就返回 None（GA 里会被判极低适应度）。

        假设，主链上 一共 5 个可替换位点（replace_sites = [12, 17, 23, 30, 42]，数字是原子索引），
        官能团库里有 10 种片段（FRAG_RANGE = range(10)，索引 0‒9）。
        genome = (-1, 3, 0, -1, 8)
        -1 → “保持原子不动”；非负整数 n → “把第 n 号官能团接到这个位点”
        对于enumerate(genome[1])，site_idx = 1, frag_idx = 3，表示：
        把官能团frag_smiles[3]，接到分子的replace_sites[1]=17位点上。
        把官能团frag_mols[frag_idx]接到分子的replace_sites[site_idx]这个位点上
        """
        mol = self.base_mol
        for site_idx, frag_idx in enumerate(genome):
            #  位点索引(site_idx), base mol 上可替换的位点为止
            #  基因值(frag_idx), 即，片段库 frag_mols 的索引
            if frag_idx == -1: # 不替换
                continue
            # 若该位点基因值为 -1，说明保留原子上的 H，不做任何化学改动；直接进入下一循环。
            mol = self.attach_fragment(
                mol, self.replace_sites[site_idx], self.frag_mols[frag_idx]
            )
            if mol is None:
                return None
        # 确保仍有两个 *
        if sum(1 for at in mol.GetAtoms() if at.GetAtomicNum() == 0) != 2:
            # 遍历新分子所有原子；统计原子序号为 0（* 哑原子）的数量；若数量不等于 2，则返回 None，视作非法个体。
            return None

        # # ==== Steric-MOD BEGIN ====
        # # 若可接触表面积超过阈值，则判为“位阻过大”，丢弃
        # if not self._steric_ok(mol):
        #     return None
        # # ==== Steric-MOD END   ====

        return Chem.MolToSmiles(mol, isomericSmiles=False)


    def random_genome(self) -> Tuple[int]:
        """
        random.random() 随机返回 0-1的数字
        random.random() < .3 代表 30% 概率做替换，70% 保持原子不动；
        若替换，则 random.choice(FRAG_RANGE) 随机挑一个官能团片段的索引；
        否则用 NO_REPLACE_ID（约定为 -1）代表“不替换”。
        """
        return tuple(random.choice(self.FRAG_RANGE) if random.random() < .3 else self.NO_REPLACE_ID for _ in range(self.GENE_LEN))

    def crossover(self, p1, p2):
        """
        c = mutate(crossover(*random.sample(elites, 2)))
        如果只有 1 个替换位点，则 cut 强制为 1；否则在 1 ~ (GENE_LEN-1) 之间随机选断点。
        p1[:cut] 取父 1 的前半段，p2[cut:] 取父 2 的后半段，拼接得到子代。
        这样子代会同时继承两个父本的特征组合
        例如，p1 (1,2,3,4) p2 (6,7,8,9) GENE_LEN = 4 cut = 1
        则 return (1,7,8,9)
        """
        cut = random.randint(1, self.GENE_LEN-1) if self.GENE_LEN > 1 else 1
        return p1[:cut] + p2[cut:]

    def mutate(self, g):
        """
        先 list(g)，因为元组不可修改。
        random.random() < MUT_RATE 以预设概率（MUT_RATE = 0.2）触发突变。
        突变的做法：随机在 “保持为 -1” 和 “任意片段索引” 之间选一个新值，覆盖原来的

        """
        g = list(g)
        for i in range(self.GENE_LEN):
            if random.random() < self.MUT_RATE:
                # FRAG_RANGE 可选官能团的总数量
                g[i] = random.choice([self.NO_REPLACE_ID, *self.FRAG_RANGE])
        return tuple(g)

    def fitness(self, genome):
        smi = self.genome_to_smiles(genome)
        if smi is None:
            return self.bad_fitness
        try:
            if self.target == 'multi':
                tg  = predict_property(smi, "Tg")
                eps = predict_property(smi, "EPS")
                # ---------- Tg 硬约束 ----------
                # if tg < self.base_tg: # 不达标直接出局
                #     return self.bad_fitness
                # if eps > self.base_eps:
                #     return self.bad_fitness
                # ---------- EPS 加权期望度 ----------
                def clip_mm(x, xmin, xmax):
                    if xmax == xmin:
                        return 0.0
                    return max(0.0, min(1.0, (x - xmin) / (xmax - xmin)))

                z_t   = clip_mm(tg,  self.tg_min,  self.tg_max)
                z_eps = clip_mm(eps, self.eps_min, self.eps_max)
                d_t   = z_t
                d_eps = 1.0 - z_eps

                w_sum   = self.w_tg + self.w_eps
                fitness = (d_t**self.w_tg * d_eps**self.w_eps) ** (1.0 / w_sum)
                return fitness
            else:
                return predict_property(smi, self.target)
        except Exception:
            return self.bad_fitness

    def plot_ga_process(self, best_hist, avg_hist, pop_fitness_hist=None):
        dark_blue = '#7A9EBA'
        dark_red = '#D28C8C'
        plt.rcParams.update({
            'font.size': 12,  # 设置默认字体大小
            'axes.titlesize': 12,  # 设置子图标题（ax.set_title）的字体大小
            'figure.titlesize': 12  # 设置整张图的标题（plt.suptitle）的字体大小
        })
        # self.N_GEN = 50 , figsize = 7
        plt.figure(figsize=(self.N_GEN * 7/50, 5))

        if pop_fitness_hist:
            # (1) 所有个体空心散点
            for g_idx, gen_fits in enumerate(pop_fitness_hist):
                plt.scatter([g_idx] * len(gen_fits),
                            gen_fits,
                            marker='o', facecolors='none',
                            edgecolors='gray', alpha=0.5, s=25,
                            label="Population" if g_idx == 0 else "")

        # (2) 最佳/平均折线
        plt.plot(best_hist, label="Best Fitness", linewidth=2, color=dark_red)
        plt.plot(avg_hist, label="Avg Fitness", linewidth=2, color=dark_blue)
        plt.xlim(0 - 1, self.N_GEN - 1 + 1)
        plt.ylim(0, 1)
        plt.xlabel("Generations", labelpad=5)
        plt.ylabel("Fitness", labelpad=5)
        plt.title("GA Search Process", size=12)
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{self.dir_folder}/ga_progress.png", dpi=600)
        plt.close()

    # def smi_to_png(self, smi, legend, png_name, size=(700, 400)):
    #     MolToImage(Chem.MolFromSmiles(smi), size,
    #                legend=legend).save(png_name)

    def smi_to_png(self, smi, legend, png_name,
                           img_size=(700, 400),
                           legend_font_size=24):

        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smi}")

        # 生成绘图对象
        drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
        opts = drawer.drawOptions()
        opts.legendFontSize = legend_font_size

        # 绘制
        drawer.DrawMolecule(mol, legend=legend or "")
        drawer.FinishDrawing()

        # 保存到文件
        out_png = Path(png_name)
        out_png.write_bytes(drawer.GetDrawingText())

    # ========= 5. GA 主循环 =========

    def run_ga(self, plot=True):
        best_g = None
        # best_tg = -1e9
        best_val = -1e9 if self.maximize else 1e9

        # --- 用于绘图的历史记录 ---
        best_hist, avg_hist = [], []
        pop_fitness_hist = []

        # --- 打开 jsonl 文件 ---
        jsonl_all   = f"{self.dir_folder}/ga_population.jsonl"        # 全部“合法”个体
        jsonl_best  = f"{self.dir_folder}/ga_best_per_gen.jsonl"      # 每代最佳
        jf_all = open(jsonl_all,  "w", encoding="utf-8")
        jf_best = open(jsonl_best, "w", encoding="utf-8")

        pop = [self.random_genome() for _ in range(self.POP_SIZE)]

        for gen in range(self.N_GEN):
            # "scored" list -> element: (genome, fitness)
            scored = [(g, self.fitness(g)) for g in pop]
            # scored.sort(key=lambda x: x[1], reverse=True)
            # 排序方向：maximize → 降序；minimize → 升序
            scored.sort(key=lambda x: x[1], reverse=self.maximize)

            # ---------- 记录 ----------

            best_hist.append(scored[0][1]) # highest fitness
            avg_hist.append(sum(f for _, f in scored) / len(scored))
            pop_fitness_hist.append([f for _, f in scored])

            # ——> 写入“本代全部合法个体”
            for g, fit in scored:
                smi = self.genome_to_smiles(g)
                if smi is not None:
                    tg_val = predict_property(smi, "Tg")  # <<< 新增
                    eps_val = predict_property(smi, "EPS")  # <<< 新增
                    json.dump({"smiles": smi,
                               "fitness": fit,
                               "generation": gen,
                               "tg": tg_val,  # <<< 新增
                               "eps": eps_val # <<< 新增
                               },
                              jf_all, ensure_ascii=False)
                    jf_all.write("\n")

            # ——> 写入“本代最佳个体”
            best_genome, best_fit = scored[0]
            best_smi_gen = self.genome_to_smiles(best_genome)
            if best_smi_gen is not None:
                best_tg = predict_property(best_smi_gen, "Tg")  # <<< 新增
                best_eps = predict_property(best_smi_gen, "EPS")  # <<< 新增
                json.dump({"smiles": best_smi_gen,
                           "fitness": best_fit,
                           "generation": gen,
                           "tg": best_tg,  # <<< 新增
                           "eps": best_eps # <<< 新增
                           },
                          jf_best, ensure_ascii=False)
                jf_best.write("\n")

            # ---------- GA 主逻辑 ----------
            # if scored[0][1] > best_val:
            #     best_g, best_tg = scored[0]
            #     print(f"[Gen {gen}] Best Tg = {best_tg:.2f} K")
            #
            # ! 当前 generation 的 best score 是否大于历史的 best score
            is_better = scored[0][1] > best_val if self.maximize else scored[0][1] < best_val
            if is_better:
                best_g, best_val = scored[0]
                # print(f"[Gen {gen}] Best Fitness = {best_val:.2f}")
            # ! 保留当前 generation 的 KEEP_TOP_K 的 genome
            elites = [g for g, _ in scored[:self.KEEP_TOP_K]]
            # if best_val > max_val: # 早停，不需要
            #     break
            next_pop = elites.copy()
            while len(next_pop) < self.POP_SIZE:
                # 交叉
                p1, p2 = random.sample(elites, 2)
                # 变异
                c = self.mutate(self.crossover(p1, p2))
                next_pop.append(c)
            # next_pop = elites + elites&crossover + elites&crossover&mutation
            pop = next_pop

        # ---------- 关闭文件 ----------
        jf_all.close()
        jf_best.close()
        # print(f"已保存 GA 全体数据  -> {jsonl_all}")
        # print(f"已保存 每代最佳数据 -> {jsonl_best}")

        # ---------- 画图 ----------
        if plot:
            self.plot_ga_process(best_hist, avg_hist, pop_fitness_hist)

        # return self.genome_to_smiles(best_g), best_val
        best_smi =  self.genome_to_smiles(best_g)
        best_prop = best_val
        # if self.target == 'multi':
        best_tg  = predict_property(best_smi, "Tg")
        best_eps = predict_property(best_smi, "EPS")
        legend  = f"Tg: {best_tg:.1f} K, k: {best_eps:.3f}"
        best_prop = (best_eps, best_tg)
        # elif self.target == 'Tg':
        #     legend = f"{self.target}: {best_val:.1f} K"
        # else:
        #     legend = f"{self.target}: {best_val:.3f}"
        self.smi_to_png(smi=self.base_smi,
                        legend = f"Tg: {self.base_tg:.1f} K, k: {self.base_eps:.3f}",
                        png_name=f"{self.dir_folder}/base_polymer_ga.png")
        self.smi_to_png(smi=best_smi,
                        legend = legend,
                        png_name=f"{self.dir_folder}/best_polymer_ga.png")
        return best_smi, best_prop


if __name__ == "__main__":
    from designPolyGA_main import frag_smiles

    N_GEN = 50
    POP_SIZE = 40
    target = 'multi'
    dir_folder = f'resultGA'
    # ========= 1. 核心骨架 & 替换位点 =========
    base_smi = "*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1"
    # base_mol = Chem.MolFromSmiles(base_smi)
    random.seed(2025)
    base_tg = predict_property(base_smi, 'Tg')
    base_eps = predict_property(base_smi, 'EPS')
    print(f"Base polymer Tg: {base_tg:.1f} K, k: {base_eps:.3f}")
    print('Running GA ...')
    GA = GeneticAlgorithm(base_smi, base_tg, base_eps,
                          frag_smiles, dir_folder, # target='multi',
                          # w_tg=1.0, w_eps=1.0,
                          N_GEN=N_GEN, POP_SIZE=POP_SIZE)
    best_smi, best_prop = GA.run_ga()

    print("=== GA 结果 ===")
    print("Best SMILES :", best_smi)
    # if target == 'multi':
    best_eps, best_tg = best_prop
    print(f'Best Tg: {best_tg:.1f} K, k: {best_eps:.3f}')
    delta_tg = (best_tg - base_tg) / base_tg * 100
    delta_eps = (best_eps - base_eps) / base_eps * 100
    print(f'Improvement ratio: Tg: {delta_tg:.2f}%, k: {delta_eps:.2f}%')

#     elif target == 'Tg':
#         print(f'Best {target}: {best_prop:.1f} K')
#         delta_tg = (best_prop - base_tg) / base_tg * 100
#         print(f'Improvement ratio: Tg: {delta_tg:.2f}%')
#     else:
#         print(f'Best {target}: {best_prop:.3f}')
#         delta_eps = (best_prop - base_eps) / base_eps * 100
#         print(f'Improvement ratio: EPS: {-delta_eps:.2f}%')
#
    plotgm = PlotGenMols(base_eps, base_tg, dir_folder)
    plotgm.plotKDEbyGenerations('tg')
    plotgm.plotKDEbyGenerations('eps')
    plotgm.plotScatterbyGenerations()
    plotgm.visualize_gen_bests(mols_per_row=5)

    # Base polymer Tg: 559.6 K, k: 3.159
    # Best SMILES : *c1c(N)cc(Oc2c(C#C)cc(-n3c(=O)c4c(C#C)c5c(=O)n(*)c(=O)c5c(C#N)c4c3=O)c(C#C)c2C#C)c(C#C)c1C#C
    # Best Tg: 636.1 K, k: 2.532
    # Improvement ratio: Tg: 13.67%, k: -19.83%

