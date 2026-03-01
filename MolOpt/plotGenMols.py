import json
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
import collections
from pathlib import Path
from io import BytesIO
from PIL import Image
# from MolOpt import sascorer
from MolOpt.calSAScore import calSAScore

class PlotGenMols:
    def __init__(self, base_eps, base_tg, dir_folder):
        # self.target = target
        self.base_eps = base_eps
        self.base_tg = base_tg
        self.dir_folder = dir_folder
        os.makedirs(self.dir_folder, exist_ok=True)
        self.jsonl_all = f"{self.dir_folder}/ga_population.jsonl" # 全部“合法”个体
        self.jsonl_best = f"{self.dir_folder}/ga_best_per_gen.jsonl" # 每代最佳
        self.jsonl_pareto_frontier = f"{self.dir_folder}/pareto_frontier.jsonl" # pareto frontier final

        records = []
        with open(self.jsonl_all, 'r', encoding='utf-8') as f:
            for line in f:
                records.append(json.loads(line))
        self.df_all = pd.DataFrame(records)

    def plotKDEbyGenerations(self, target):
        """
        target = 'tg' / 'eps'
        """
        out_png = f'{self.dir_folder}/kde_by_gen_{target}.png'
        plt.rcParams.update({
            'font.size': 12,  # 设置默认字体大小
            'axes.titlesize': 12,  # 设置子图标题（ax.set_title）的字体大小
            'figure.titlesize': 12  # 设置整张图的标题（plt.suptitle）的字体大小
        })
        gens = sorted(self.df_all['generation'].unique())
        norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))
        cmap = mpl.colormaps['viridis']  # ← 避免 get_cmap 的弃用警告
        line_kw = dict(linewidth=1.2, alpha=0.8)
        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])  # 供 colorbar 使用

        fig, ax = plt.subplots(figsize=(7, 5))
        for g in gens:
            subset = self.df_all.loc[self.df_all['generation'] == g, target]
            if subset.nunique() > 1:
                subset.plot(kind='kde',
                            ax=ax,
                            color=cmap(norm(g)),
                            **line_kw)
        if target == 'tg':
            base_value = self.base_tg
        else:
            base_value = self.base_eps
        line = ax.axvline(x=base_value, color='black', linestyle='--', linewidth=1.5, label='Base Value')
        if target == 'eps':
            ax.set_title(r'GA Population: $k$ by Generation')
            ax.set_xlabel(r'$k$', labelpad=5)
        if target == 'tg':
            ax.set_title(r'GA Population: $T_g$ by Generation')
            ax.set_xlabel(r'$T_g$ (K)', labelpad=5)
        ax.set_ylabel('Density', labelpad=5)
        cbar = fig.colorbar(sm, ax=ax, pad=0.02)  # ← 指定 ax，避免找不到轴
        cbar.set_label('Generation')
        # ax.legend()
        ax.legend(handles=[line], loc='upper right')
        fig.tight_layout()
        fig.savefig(out_png, dpi=600)
        plt.close(fig)

    def plotScatterbyGenerations(self):

        out_png = f'{self.dir_folder}/scatter_by_gen.png'
        plt.rcParams.update({
            'font.size': 12,  # 设置默认字体大小
            'axes.titlesize': 12,  # 设置子图标题（ax.set_title）的字体大小
            'figure.titlesize': 12  # 设置整张图的标题（plt.suptitle）的字体大小
        })

        plt.figure(figsize=(7, 5))
        # 以 generation 做颜色映射，generation 越大颜色越深
        scatter = plt.scatter(self.df_all['eps'], self.df_all['tg'],
                              c=self.df_all['generation'],
                              cmap='viridis',  # 深浅渐变
                              # s = 36, default
                              alpha=0.8)
        # start point (base_esp, base_tg) 用红色五角星标注

        plt.scatter(self.base_eps, self.base_tg,
                    color='red', marker='*', s=100,
                    label='Base Molecule') # plt.legend(fontsize=10)设置图例字号

        #
        plt.xlabel(r'$k$', labelpad=5)
        plt.ylabel(r'$T_g$ (K)', labelpad=5)
        plt.title(r'GA Population: $k$ vs $T_g$ by Generation')
        cbar = plt.colorbar(scatter)
        cbar.set_label('Generation')
        plt.tight_layout()
        plt.legend(loc='lower left')  # 放在右下角 —— upper/lower right/left
        # plt.legend()
        # bbox_inches='tight'
        plt.savefig(out_png, dpi=600)
        plt.close()

    def visualize_gen_bests(self, mols_per_row=4, img_size=(800*3, 500*3), legend_font_size=48,):
        """
        可视化各代唯一最佳分子，支持设置 legend 字体大小
        """
        out_png = f"{self.dir_folder}/gen_bests.png"
        with open(self.jsonl_best, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        uniq = collections.OrderedDict()
        for rec in records:
            smi = rec["smiles"]
            if smi not in uniq:
                uniq[smi] = (rec["generation"], rec["tg"], rec["eps"])

        mols, legends = [], []
        for smi, (gen, tg, eps) in uniq.items():
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mols.append(m)
            legends.append(f"Generation: {gen}, Tg: {tg:.1f} K, k: {eps:.3f}")

        if mols:
            # 计算行数
            n_mols = len(mols)
            n_cols = mols_per_row
            n_rows = (n_mols + n_cols - 1) // n_cols

            drawer = rdMolDraw2D.MolDraw2DCairo(n_cols * img_size[0] // mols_per_row,
                                                n_rows * img_size[1] // mols_per_row,
                                                img_size[0] // mols_per_row,
                                                img_size[1] // mols_per_row)
            draw_opts = drawer.drawOptions()
            draw_opts.legendFontSize = legend_font_size  # 设置 legend 字体大小

            drawer.DrawMolecules(mols, legends=legends)
            drawer.FinishDrawing()

            img = Image.open(BytesIO(drawer.GetDrawingText()))
            img.save(out_png)
        else:
            print("[WARN] 未找到可用分子，未生成图像")

    def visualize_pareto_frontier(self, mols_per_row=4, img_size=(800*3, 500*3), legend_font_size=48,):
        """
        可视化各代唯一最佳分子，支持设置 legend 字体大小
        """
        out_png = f"{self.dir_folder}/pareto_frontier_final.png"
        with open(self.jsonl_pareto_frontier, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f]

        uniq = collections.OrderedDict()
        for rec in records:
            smi = rec["smiles"]
            sa_score = calSAScore(smi)
            if smi not in uniq:
                # uniq[smi] = (rec["generation"], rec["tg"], rec["eps"])
                uniq[smi] = (rec["tg"], rec["eps"], sa_score)

        # 按 sa_score 升序排序 uniq
        # uniq_list = sorted(uniq.items(), key=lambda x: x[1][2])
        # 按 k 升序排序 uniq # ———— 25-12-15修改
        uniq_list = sorted(uniq.items(), key=lambda x: x[1][1])
        # rec["tg"]（0）, rec["eps"]（1）, sa_score（2）

        # uniq: dict {
        #       "smiles":[tg, eps, sascore],
        #       }
        # uniq_list: list [
        #       ["smiles":[tg, eps, sascore]],
        #       ]

        mols, legends = [], []
        # for smi, (gen, tg, eps) in uniq.items():
        for smi, (tg, eps, sa_score) in uniq_list:
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
            mols.append(m)
            legends.append(f"Tg: {tg:.1f} K, k: {eps:.3f}, SAScore: {sa_score:.2f}")

        if mols:
            # 计算行数
            n_mols = len(mols)
            n_cols = mols_per_row
            n_rows = (n_mols + n_cols - 1) // n_cols

            drawer = rdMolDraw2D.MolDraw2DCairo(n_cols * img_size[0] // mols_per_row,
                                                n_rows * img_size[1] // mols_per_row,
                                                img_size[0] // mols_per_row,
                                                img_size[1] // mols_per_row)
            draw_opts = drawer.drawOptions()
            draw_opts.legendFontSize = legend_font_size  # 设置 legend 字体大小

            drawer.DrawMolecules(mols, legends=legends)
            drawer.FinishDrawing()

            img = Image.open(BytesIO(drawer.GetDrawingText()))
            img.save(out_png)
        else:
            print("[WARN] 未找到可用分子，未生成图像")



# def smi_to_png(self, smi, legend, png_name, size=(700, 400)):
#     MolToImage(Chem.MolFromSmiles(smi), size,
#                legend=legend).save(png_name)
# def draw_single_smiles(smi, legend, png_name,
#                        img_size = (400, 300),
#                        legend_font_size = 24):
#
#     mol = Chem.MolFromSmiles(smi)
#     if mol is None:
#         raise ValueError(f"Invalid SMILES: {smi}")
#
#     # 生成绘图对象
#     drawer = rdMolDraw2D.MolDraw2DCairo(img_size[0], img_size[1])
#     opts = drawer.drawOptions()
#     opts.legendFontSize = legend_font_size
#
#     # 绘制
#     drawer.DrawMolecule(mol, legend=legend or "")
#     drawer.FinishDrawing()
#
#     # 保存到文件
#     out_png = Path(png_name)
#     out_png.write_bytes(drawer.GetDrawingText())

# if __name__ == '__main__':
#     target = 'multi'
#     plotgm = PlotGenMols(target)
#     plotgm.plotKDEbyGenerations('tg')
#     plotgm.plotKDEbyGenerations('eps')
#     plotgm.plotScatterbyGenerations()


def visualize_base_polymers(dir_folder, records,
                            mols_per_row=4, img_size=(800*3, 500*3),
                            legend_font_size=48,
                            ):
    """
    可视化各代唯一最佳分子，支持设置 legend 字体大小
    """
    out_png = f"{dir_folder}/base_polymers.png"
    # with open(self.jsonl_best, "r", encoding="utf-8") as f:
    #     records = [json.loads(line) for line in f]
    # {"smiles": "*c1c(C#N)c(F)c(Oc2ccc(-n3c(=O)c4c(C#N)c5c(=O)n(*)c(=O)c5c(C#N)c4c3=O)c(C#N)c2C#N)c(C#N)c1C#N",
    # "tg": 534.9876098632812,
    # "eps": 2.127666473388672,
    # "obj": [2.127666473388672, -534.9876098632812],
    # "rank": 0}
    uniq = collections.OrderedDict()
    for rec in records:
        smi = rec["smiles"]
        if smi not in uniq:
            # uniq[smi] = (rec["generation"], rec["tg"], rec["eps"])
            uniq[smi] = (rec["id"], rec["tg"], rec["eps"], rec["sa"])

    mols, legends = [], []
    for smi, (id, tg, eps, sa) in uniq.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        mols.append(m)
        legends.append(f"ID: {id}, Tg: {tg:.1f} K, k: {eps:.3f}, SAScore: {sa:.2f}")

    if mols:
        # 计算行数
        n_mols = len(mols)
        n_cols = mols_per_row
        n_rows = (n_mols + n_cols - 1) // n_cols

        drawer = rdMolDraw2D.MolDraw2DCairo(n_cols * img_size[0] // mols_per_row,
                                            n_rows * img_size[1] // mols_per_row,
                                            img_size[0] // mols_per_row,
                                            img_size[1] // mols_per_row)
        draw_opts = drawer.drawOptions()
        draw_opts.legendFontSize = legend_font_size  # 设置 legend 字体大小

        drawer.DrawMolecules(mols, legends=legends)
        drawer.FinishDrawing()

        img = Image.open(BytesIO(drawer.GetDrawingText()))
        img.save(out_png)
    else:
        print("[WARN] 未找到可用分子，未生成图像")

