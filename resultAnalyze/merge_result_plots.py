import matplotlib.pyplot as plt

import os
from pathlib import Path

def combine_metric_trend_images(main_dir: str,
                                start_id: int = 0,
                                end_id: int = 10,
                                dpi: int = 200) -> str:
    """
    将指定 id 范围内的三类指标图合并为一张大图。

    每一行对应一个 id，三列依次为：
    1. Hypervolume
    2. Pareto Coverage
    3. Spacing

    start_id 和 end_id 均包含在合并范围内，例如：
    start_id=0, end_id=5 表示合并 id0 至 id5，共 6 行。
    """
    if start_id < 0:
        raise ValueError("start_id 不能小于 0。")
    if end_id < start_id:
        raise ValueError("end_id 必须大于或等于 start_id。")

    id_list = list(range(start_id, end_id + 1))
    image_info = [
        ("metric_hypervolume_trend.png", "Hypervolume"),
        ("metric_pareto_coverage_trend.png", "Pareto Coverage"),
        ("metric_spacing_trend.png", "Spacing"),
    ]

    # squeeze=False 可保证即使只选择一个 id，axes 仍然保持二维数组形式。
    fig, axes = plt.subplots(
        nrows=len(id_list),
        ncols=3,
        figsize=(15, 3.2 * len(id_list)),
        squeeze=False,
    )

    missing_images = []
    for row_index, current_id in enumerate(id_list):
        metrics_dir = Path(main_dir) / f"metrics_id{current_id}"

        for column_index, (image_name, column_title) in enumerate(image_info):
            ax = axes[row_index, column_index]
            image_path = metrics_dir / image_name

            if image_path.exists():
                image = plt.imread(image_path)
                ax.imshow(image)
            else:
                missing_images.append(str(image_path))
                ax.text(
                    0.5, 0.5,
                    f"Image not found:\n{image_name}",
                    ha="center", va="center",
                    fontsize=10,
                    transform=ax.transAxes,
                )

            ax.axis("off")

            # 只在第一行显示三列的标题，避免重复占用空间。
            # if row_index == 0:
            #     ax.set_title(column_title, fontsize=14, pad=10)

        # 在每行最左侧标注当前 id。
        axes[row_index, 0].text(
            -0.03, 0.5,
            f"ID {current_id}",
            ha="right", va="center",
            fontsize=13,
            fontweight="bold",
            transform=axes[row_index, 0].transAxes,
        )

    fig.tight_layout(pad=1.0)

    output_path = (
        Path(main_dir)
        / f"metric_trends_combined_id{start_id}_to_id{end_id}.png"
    )
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    print(f"Combined metric image saved to: {output_path}")
    if missing_images:
        print("Warning: the following images were not found:")
        for image_path in missing_images:
            print(f"  - {image_path}")

    return str(output_path)
#=====【结果图合并】===== 结束


if __name__ == "__main__":
    RANDOM_SEED = "2026"
    main_dir = f'resultCompareSeed{RANDOM_SEED}'

    MERGE_START_ID = 0   # 合并起始 id（包含）
    MERGE_END_ID = 5    # 合并结束 id（包含）；例如改为 5，即合并 id0~id5

    # MERGE_START_ID = 0   # 合并起始 id（包含）
    # MERGE_END_ID = 10    # 合并结束 id（包含）；例如改为 5，即合并 id0~id5

    # 所有 id 的三类指标图生成完成后，再执行总图合并。
    combine_metric_trend_images(
        main_dir=main_dir,
        start_id=MERGE_START_ID,
        end_id=MERGE_END_ID,
    )

