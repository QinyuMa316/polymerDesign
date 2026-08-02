import json
import pandas as pd
import numpy as np
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
import seaborn as sns


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
            'font.family': 'Arial',
            'axes.titlesize': 15,  # 主标题
            'axes.labelsize': 15,  # 横纵轴标题
            'xtick.labelsize': 13,  # x刻度
            'ytick.labelsize': 13  # y刻度
        })
        gens = sorted(self.df_all['generation'].unique())
        norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))
        cmap = mpl.colormaps['viridis']  # ← 避免 get_cmap 的弃用警告
        line_kw = dict(linewidth=1.2, alpha=0.7)
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
            'font.family': 'Arial',
            'axes.titlesize': 15,  # 主标题
            'axes.labelsize': 15,  # 横纵轴标题
            'xtick.labelsize': 13,  # x刻度
            'ytick.labelsize': 13  # y刻度
        })

        plt.figure(figsize=(7, 5))
        # 以 generation 做颜色映射，generation 越大颜色越深
        scatter = plt.scatter(self.df_all['eps'], self.df_all['tg'],
                              c=self.df_all['generation'],
                              cmap='viridis',  # 深浅渐变
                              # s = 36, default
                              alpha=0.7)
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

        # print("=== sorted uniq_list by eps ===")
        # for i, (smi, (tg, eps, sa_score)) in enumerate(uniq_list[:10]):
        #     print(i, f"tg={tg:.1f}, k={eps:.3f}, sa={sa_score:.2f}")

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

    def plotScatterWithKDEbyGenerations(self, id, version):
        """
        一张图同时显示：
        - 中间：eps vs tg 散点图（按 generation 着色）
        - 上方：eps 的 KDE 分布（按 generation 着色）
        - 右侧：tg 的 KDE 分布（按 generation 着色）
        """

        out_png = f'{self.dir_folder}/scatter_kde_by_gen.png'

        plt.rcParams.update({
            'font.family': 'Arial',
            'axes.titlesize': 15,
            'axes.labelsize': 15,
            'xtick.labelsize': 13,
            'ytick.labelsize': 13
        })

        df = self.df_all[['eps', 'tg', 'generation']].dropna().copy()
        df = df.sort_values('generation')

        gens = sorted(df['generation'].unique())
        norm = mpl.colors.Normalize(vmin=min(gens), vmax=max(gens))
        cmap = mpl.colormaps['viridis']
        palette = {g: cmap(norm(g)) for g in gens}

        # ===== 手动控制主图范围 =====
        x = df['eps'].to_numpy()
        y = df['tg'].to_numpy()

        x_min, x_max = np.min(x), np.max(x)
        y_min, y_max = np.min(y), np.max(y)

        x_pad = 0.3 * (x_max - x_min)
        y_pad = 0.3 * (y_max - y_min)

        xlim = (x_min - x_pad, x_max + x_pad)
        ylim = (y_min - y_pad, y_max + y_pad)

        # ===== 建图 =====
        g = sns.JointGrid(
            data=df,
            x='eps',
            y='tg',
            height=8,
            ratio=4,
            space=0.08
        )
        g.figure.set_size_inches(8, 6)

        # ===== 主散点图 =====
        sns.scatterplot(
            data=df,
            x='eps',
            y='tg',
            hue='generation',
            palette=palette,
            s=50,
            alpha=0.7,
            edgecolor=None,
            legend=False,
            ax=g.ax_joint
        )

        # ===== 上方边际 KDE =====
        sns.kdeplot(
            data=df,
            x='eps',
            hue='generation',
            hue_order=gens,
            palette=palette,
            common_norm=False,
            fill=False,
            linewidth=1.1,
            alpha=0.7,
            legend=False,
            ax=g.ax_marg_x
        )

        # ===== 右侧边际 KDE =====
        sns.kdeplot(
            data=df,
            y='tg',
            hue='generation',
            hue_order=gens,
            palette=palette,
            common_norm=False,
            fill=False,
            linewidth=1.1,
            alpha=0.7,
            legend=False,
            ax=g.ax_marg_y
        )

        # ===== Base molecule =====
        g.ax_joint.scatter(
            self.base_eps, self.base_tg,
            color='red', marker='*', s=180,
            label='Base Molecule', zorder=10
        )

        # ===== 坐标范围：主图和边际图同步 =====
        g.ax_joint.set_xlim(xlim)
        g.ax_joint.set_ylim(ylim)
        g.ax_marg_x.set_xlim(xlim)  # 上方边际图的 x 范围与主图一致
        g.ax_marg_y.set_ylim(ylim)  # 右侧边际图的 y 范围与主图一致

        # ===== 标签 =====
        g.ax_joint.set_xlabel(r'$k$')
        g.ax_joint.set_ylabel(r'$T_g$ (K)')
        g.ax_marg_x.set_ylabel('Density')
        g.ax_marg_y.set_xlabel('Density')

        # ===== 图例 =====
        g.ax_joint.legend(loc='lower left'
                          # , fontsize=11, frameon=True
                          )

        # ===== 标题 =====
        g.figure.suptitle(
            f'{version}'+r' Population: $k$ vs $T_g$ by Generation' + f' (id: {id})',
            y=0.98,
            # fontsize=20
        )

        # ===== 调整布局，为 colorbar 留出空间 =====
        g.figure.subplots_adjust(top=0.92, right=0.86)  # 右侧预留 0.14 宽度

        # ===== 【修改】动态计算 colorbar 位置，使其与右侧边际图对齐 =====
        # 获取右侧边际图的位置 [left, bottom, width, height]
        pos_marg_y = g.ax_marg_y.get_position()
        # 在边际图右侧创建 colorbar 轴：x0 = 边际图右边界 + 小间距，宽度固定，y0 和 height 与边际图相同
        cax_width = 0.03
        cax_gap = 0.01
        cax = g.figure.add_axes([
            pos_marg_y.x1 + cax_gap,  # left
            pos_marg_y.y0,  # bottom
            cax_width,  # width
            pos_marg_y.height  # height 与边际图一致
        ])

        sm = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = g.figure.colorbar(sm, cax=cax)
        cbar.set_label('Generation',
                       # fontsize=12
                       )
        # 可选：调整 colorbar 刻度朝向，减少占用空间
        cbar.ax.tick_params(direction='in', length=4)

        # ===== 美化 =====
        sns.despine(fig=g.figure)
        g.figure.savefig(out_png, dpi=600, bbox_inches='tight')
        plt.close(g.figure)




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

