import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


def draw(pso, parameter: dict) -> None:
    # 初始化
    Tm = np.zeros(parameter["fjsp"]["machine"]["size"])  # 每一機台的結束時間
    To = np.zeros(
        (parameter["fjsp"]["job"]["size"], max(parameter["fjsp"]["opra"]["seq"]))
    )  # 每一工件每一製程的結束時間

    OS = pso.decoding_OS(particle=pso.gbest[1])
    MS = pso.decoding_MS(particle=pso.gbest[1])
    gantt = list()
    for job, opra in OS:
        machine = MS[(job, opra)]
        length = parameter["data"]["regular"][job][opra][machine]  # 工時
        # 若為第一道製程，則被選中機台的結束時間直接加上工時
        if opra == 0:
            start = 0
            Tm[machine] += length
            To[job, opra] = Tm[machine]
            end = Tm[machine]
        # 若不為第一道製程，則機台結束時間為 max(前一製程結束時間, 被選中機台結束時間) + 工時
        else:
            start = max(To[job, opra - 1], Tm[machine])
            Tm[machine] = max(To[job, opra - 1], Tm[machine]) + length
            To[job, opra] = Tm[machine]
            end = Tm[machine]
        gantt.append({"Job": job, "Machine": machine, "Start": start, "End": end})
    gantt = pd.DataFrame(gantt)

    colors = [
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#d62728",
        "#9467bd",
        "#8c564b",
        "#e377c2",
        "#7f7f7f",
        "#bcbd22",
        "#17becf",
        "#393b79",
        "#5254a3",
        "#6b6ecf",
        "#9c9ede",
        "#637939",
        "#8ca252",
        "#b5cf6b",
        "#cedb9c",
        "#8c6d31",
        "#bd9e39",
        "#e7ba52",
        "#e7cb94",
        "#843c39",
        "#ad494a",
        "#d6616b",
        "#e7969c",
        "#7b4173",
        "#a55194",
        "#ce6dbd",
        "#de9ed6",
        "#7f3b08",
        "#b35806",
        "#e08214",
        "#fdb863",
        "#843c39",
        "#d9d9d9",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#08519c",
        "#9ecae1",
        "#6baed6",
        "#4292c6",
        "#2171b5",
        "#084594",
        "#d9d9d9",
        "#c6dbef",
        "#9ecae1",
        "#6baed6",
    ]

    sorted_labels = sorted(gantt["Job"].unique())
    slected_colors = np.random.choice(colors, len(sorted_labels), replace=False)
    color_dict = {Job: Color for Job, Color in zip(sorted_labels, slected_colors)}

    # 圖片打印
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 設定字體，避免中文字亂碼
    gs = GridSpec(2, 2)
    ax1 = plt.subplot(gs[0, 0])
    ax1.set_title("全局最优解的变化情况")
    ax1.plot(pso.pso_base)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("每次迭代后种群适应度最小值的变化情况")
    ax2.plot(pso.iter_process)
    ax3 = plt.subplot(gs[1, :])
    ax3.set_title("甘特圖")
    for index, row in gantt.iterrows():
        ax3.barh(
            row["Machine"],
            width=row["End"] - row["Start"],
            left=row["Start"],
            color=color_dict.get(row["Job"], "gray"),
        )
    ax3.set_yticks(gantt["Machine"])
    ax3.set_yticklabels(gantt["Machine"])
    ax3.set_ylabel("Machine")
    ax3.set_xlabel("Timeline")
    handles = [
        plt.Rectangle((0, 0), 1, 1, fc=color_dict.get(Job, "gray"))
        for Job in sorted_labels
    ]
    ax3.legend(
        handles, sorted_labels, title="Job", loc="center left", bbox_to_anchor=(1, 0.5)
    )
    plt.show()
