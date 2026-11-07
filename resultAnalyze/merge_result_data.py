import pandas as pd

if __name__ == "__main__":


    seed_list = ["2025_", "3139", "3710", "6095", "6227", "2026"]
    # "1501"
    cnt = str(len(seed_list))
    all_df_path = f"opt_results_all_{cnt}-3.csv"
    result_df_path = f"opt_results_mean_std_{cnt}-3.csv"

    id_column = "Base Polymer SMILES"

    metric_columns = [
        "GA HV",
        "NSGA-II HV",
        "C(NSGA-II, GA)",
        "C(GA, NSGA-II)",
        "GA Spacing",
        "NSGA-II Spacing",
    ]

    selected_columns = [id_column] + metric_columns

    df_list = []

    for seed in seed_list:
        opt_results_path = (
            f"resultCompareSeed{seed}/opt_results.csv"
        )

        df = pd.read_csv(
            opt_results_path,
            usecols=selected_columns
        )

        # 记录每条数据对应的实验种子
        df["seed"] = seed

        df_list.append(df)

    # 合并5个表格
    all_df = pd.concat(
        df_list,
        ignore_index=True
    )

    # 按SMILES第一次出现的顺序生成固定ID
    unique_smiles = (
        all_df[id_column]
        .drop_duplicates()
        .tolist()
    )

    smiles_to_id = {
        smiles: idx
        for idx, smiles in enumerate(unique_smiles)
    }

    # 将ID插入到第一列
    all_df.insert(
        0,
        "ID",
        all_df[id_column].map(smiles_to_id)
    )

    # 确保指标列为数值类型
    all_df[metric_columns] = (
        all_df[metric_columns]
        .apply(
            pd.to_numeric,
            errors="coerce"
        )
    )


    # 保存带ID的所有原始结果
    all_df.to_csv(
        all_df_path,
        index=False,
        encoding="utf-8-sig"
    )

    # 按ID和SMILES分组
    group_columns = [
        "ID",
        id_column
    ]

    # 计算均值
    mean_df = (
        all_df
        .groupby(
            group_columns,
            sort=False
        )[metric_columns]
        .mean()
    )

    # 计算样本标准差，ddof=1
    std_df = (
        all_df
        .groupby(
            group_columns,
            sort=False
        )[metric_columns]
        .std()
    )

    # 格式化为 mean±std，均保留两位小数
    result_df = mean_df.copy()

    for column in metric_columns:
        result_df[column] = [
            (
                f"{mean:.2f}±{std:.2f}"
                if pd.notna(std)
                else f"{mean:.2f}±0.00"
            )
            for mean, std in zip(
                mean_df[column],
                std_df[column]
            )
        ]

    # 将ID和Base Polymer SMILES恢复为普通列
    result_df = result_df.reset_index()

    # 按ID排序
    result_df = (
        result_df
        .sort_values("ID")
        .reset_index(drop=True)
    )

    # 保存均值和标准差结果
    result_df.to_csv(
        result_df_path,
        index=False,
        encoding="utf-8-sig"
    )

    # print("已保存带ID的原始数据：opt_results_all.csv")
    # print("已保存带ID的均值和标准差：opt_results_mean_std.csv")