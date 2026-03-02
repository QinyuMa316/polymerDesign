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


if __name__ == "__main__":

    N_GEN = 50
    POP_SIZE = 40
    # target = 'multi'
    random.seed(2025)
    version = 'GA'
    # version = 'NSGA2'

    main_dir = f'result{version}'
    if 'NS' not in version:
        from designPolyGA import GeneticAlgorithm

    elif 'NS' in version:
        from designPolyNSGA2 import NSGA2

    # ========= 1. 核心骨架 & 替换位点 =========
    df = pd.read_csv('data_mod/polyimides_pred_prop_base_str_std.csv')
    scaffolds = df['scaffold'].tolist()
    scaffolds_unique = list(dict.fromkeys(scaffolds))
    print(len(scaffolds), "->", len(scaffolds_unique))
    # 12 -> 11
    base_polymers = []
    # [
    #  {"id":xx, "smiles":xxx, "tg":xx, "eps":xx, "sa":xx},
    #  {....}
    # ]
    opt_results = []
    for id, scaffold in tqdm(enumerate(scaffolds_unique)):
        print('\n'+'='*50)
        print(f'id: {id}, Base polymer: {scaffold}')
        # core_smiles = "*c1ccc(Oc2ccc(-n3c(=O)c4cc5c(=O)n(*)c(=O)c5cc4c3=O)cc2)cc1"
        base_smi = scaffold
        dir_folder = f'{main_dir}/{version}_id{id}'
        # base_mol = Chem.MolFromSmiles(base_smi)
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
        if 'NS' not in version:
            print('Running GA ...')
            GA = GeneticAlgorithm(base_smi,
                                  base_tg, base_eps,
                                  frag_smiles, dir_folder,
                                  # w_tg=1.0, w_eps=1.0,
                                  N_GEN = N_GEN, POP_SIZE = POP_SIZE)
            best_smi, best_prop = GA.run_ga()
            best_eps, best_tg = best_prop
            print("Best SMILES :", best_smi)
            print(f'Best Tg: {best_tg:.1f} K, k: {best_eps:.3f}')
            delta_tg = (best_tg - base_tg) / base_tg * 100
            delta_eps = (best_eps - base_eps) / base_eps * 100
            print(f'Change Ratio: Tg: {delta_tg:.2f}%, EPS: {delta_eps:.2f}%')

        elif 'NS' in version:
            print('Running NSGA-II ...')
            GA = NSGA2(base_smi, base_tg, base_eps,
                       frag_smiles, dir_folder, N_GEN, POP_SIZE)
            pareto_solutions = GA.run_ga(
                plot=True,
                # debug_select=True,      # True 会逐次打印 tournament 选择细节（输出很多）
                # save_parent_viz=True     # True 会每代输出 Pareto前沿+父母选择图
            )
            # print("=== NSGA-II Pareto Frontier ===")
            frontier_size = len(pareto_solutions)


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
                # print(f"ID: {i} Tg: {cur_tg:.1f} K, k: {cur_eps:.2f}, SAScore: {cur_sa:.2f}")
                # print(f"SMILES: {sol['smiles']}")
                # print(f'Improvement ratio: Tg: {delta_tg:.2f}%, k: {delta_eps:.2f}%')
            avg_tg = sum(delta_tg_list) / len(delta_tg_list)
            avg_eps = sum(delta_eps_list) / len(delta_eps_list)
            avg_sa = sum(sa_list) / len(sa_list)
            delta_avg_sa = avg_sa - base_sa

            print(f"Base polymer Tg: {base_tg:.1f} K, k: {base_eps:.3f}, SAScore: {base_sa:.2f}")
            print(f"Frontier size = {frontier_size}")
            print(f"Avg Tg improvement: {avg_tg:.2f}%")
            print(f"Avg k decrease: {avg_eps:.2f}%")
            print(f"Avg SAScore: {avg_sa:.2f}, Delta Avg SAScore: {delta_avg_sa:.2f}")

            opt_results.append({
                "Base Polymer SMILES": base_smi,
                "Tg": round(base_tg, 1),
                "k": round(base_eps, 3),
                "SAScore": round(base_sa, 2),
                "Frontier Size": frontier_size,
                "Avg Tg improvement (%)": round(avg_tg, 2),
                "Avg k decrease (%)": round(avg_eps, 2),
                "Avg SAScore": round(avg_sa, 2),
                "Avg Delta SAScore": round(delta_avg_sa, 2),
            })

        plotgm = PlotGenMols(base_eps, base_tg, dir_folder)
        plotgm.plotKDEbyGenerations('tg')
        plotgm.plotKDEbyGenerations('eps')
        plotgm.plotScatterbyGenerations()
        plotgm.visualize_gen_bests(mols_per_row=4)
        if 'NS' in version:
            plotgm.visualize_pareto_frontier(mols_per_row=5)
        # print('\n\n')

    opt_results_df = pd.DataFrame(opt_results) # list[dict, dict,]
    opt_results_df.to_csv(f'{main_dir}/opt_results.csv')
    base_polymers_df = pd.DataFrame(base_polymers) # list[dict, dict,]
    base_polymers_df.to_csv(f'{main_dir}/base_polymers.csv')
    visualize_base_polymers(main_dir, base_polymers, mols_per_row=4)

