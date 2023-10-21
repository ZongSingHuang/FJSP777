# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:05:23 2019

@author: david
"""
import os
import time
from operator import itemgetter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from addict import Dict
from matplotlib.gridspec import GridSpec

"""
改变问题规模时需要改动的参数：工件参数workpiece，工序数目process，机器数目machine，
机器选择部分的范围上限，迭代次数（规模小的时候可能降低迭代次数）
"""


# 建立絕對路徑
def get_abs_path(parameter: dict) -> dict:
    # 複製
    para_path = parameter["path"]

    # 取得當前路徑
    cwd = os.getcwd()

    # 建立 rawdata 的絕對路徑
    para_path["rawdata"] = os.path.join(
        cwd, para_path["rawdata"], para_path["filename"]
    )

    # 建立 result 的絕對路徑
    para_path["result"] = os.path.join(cwd, para_path["result"])
    return para_path


# 檢查資料夾和檔案是否存在
def check_dir_exist(parameter: dict) -> None:
    # 複製
    para_path = parameter["path"]

    if not os.path.exists(para_path["rawdata"]):
        raise FileNotFoundError(f"{para_path['rawdata']} 不存在!")

    if not os.path.exists(para_path["result"]):
        os.makedirs(parameter["result"], exist_ok=True)


# 移除不可使用的機台
def regular_data(parameter: dict) -> dict:
    # 複製
    raw = parameter["data"]["raw"].copy()

    # 初始化
    data = Dict()

    # by normal
    for (job, opra), row in raw.iterrows():
        # 新增
        data[job][opra] = {k: v for k, v in row.items() if v != "-"}
    return data


# 將工時歸一化後進行排序，最後取累積和
def normalize_data(parameter: dict) -> dict:
    # 複製
    raw = parameter["data"]["raw"].copy()

    # 將不可用機台的工時，以 na 取代
    raw.replace("-", pd.NA, inplace=True)

    # 初始化
    normal = Dict()

    # by normal
    for (job, opra), row in raw.iterrows():
        # 取倒數
        reciprocal = 1 / row
        # 倒數加總
        sigma = reciprocal.sum()
        # 歸一化
        normalization = reciprocal / sigma
        # 排序
        normalization.sort_values(inplace=True)
        # 累積和
        cumsum = normalization.cumsum()
        # 新增
        normal[job][opra] = {k: v for k, v in cumsum.items() if pd.notna(v)}
    return normal


# 取得排程參數
def get_fjsp_parameter(parameter: dict) -> dict:
    # 複製
    data = parameter["data"]

    # 初始化
    fjsp = Dict()

    # 機台數目
    fjsp["machine"]["size"] = data["raw"].columns.size
    fjsp["machine"]["seq"] = data["raw"].columns.tolist()

    # 工件數目
    fjsp["job"]["size"] = len(data["regular"].keys())
    fjsp["job"]["seq"] = list(data["regular"].keys())

    # 製程數目
    fjsp["opra"]["seq"] = [len(v) for v in data["regular"].values()]
    fjsp["opra"]["size"] = sum(fjsp["opra"]["seq"])
    return fjsp


# 初始化種族、速度和計算適應值
def initial_population(parameter: dict) -> tuple:
    # 複製
    para_alg = parameter["alg"]
    para_fjsp = parameter["fjsp"]
    para_data = parameter["data"]

    # 初始化種族(前半為投料順序，後半為加工機台)
    pop = np.zeros((para_alg["popsize"], para_fjsp["opra"]["size"] * 2))

    # 初始化速率(前半為投料順序，後半為加工機台)
    vel = np.zeros_like(pop)

    # 初始化適應值
    fitness = np.zeros(para_alg["popsize"])

    # 逐粒子
    for idx in range(para_alg["popsize"]):
        # 取得粒子
        particle = pop[idx]

        # 隨機生成投料順序
        os = list()
        for job, opra_size in zip(para_fjsp["job"]["seq"], para_fjsp["opra"]["seq"]):
            os += [job] * opra_size
        particle[: para_fjsp["opra"]["size"]] = os
        np.random.shuffle(particle[: para_fjsp["opra"]["size"]])

        # 隨機生成加工機台
        ms = list()
        for job, opra_machine_len in para_data["regular"].items():
            for opra, machine_len in opra_machine_len.items():
                machine = list(machine_len.keys())
                ms.append(np.random.choice(machine))
        particle[para_fjsp["opra"]["size"] :] = ms

        # 計算粒子的適應值(總完工時間)
        fitness[idx] = calculate(particle=particle, parameter=parameter)
    return (pop, vel, fitness)


# 計算粒子的適應值(總完工時間)
def calculate(particle: np.array, parameter: dict):
    # 複製
    para_fjsp = parameter["fjsp"]
    data = parameter["data"]["regular"]

    # 初始化
    Tm = np.zeros(para_fjsp["machine"]["size"])  # 每一機台的結束時間
    To = np.zeros(
        (para_fjsp["job"]["size"], max(para_fjsp["opra"]["seq"]))
    )  # 每一工件每一製程的結束時間

    # 投料順序(工件編號, 製程編號)
    OS = decoding_OS(particle=particle, parameter=parameter)
    MS = decoding_MS(particle=particle, parameter=parameter)

    for os_ in OS:
        # 取得工時
        job = os_[0]  # 工件編號
        opra = os_[1]  # 製程編號
        machine = MS[os_]  # 機台編號
        length = data[job][opra][machine]  # 工時

        # 若為第一道製程，則被選中機台的結束時間直接加上工時
        if opra == 0:
            Tm[machine] += length
            To[job, opra] = Tm[machine]
        # 若不為第一道製程，則機台結束時間為 max(前一製程結束時間, 被選中機台結束時間) + 工時
        else:
            Tm[machine] = max(To[job, opra - 1], Tm[machine]) + length
            To[job, opra] = Tm[machine]
    return max(Tm)


# 將 OS 解譯為帶有(工件編號, 製程編號)的投料順序
def decoding_OS(particle: np.array, parameter: dict) -> list:
    # 複製
    para_fjsp = parameter["fjsp"]

    # 初始化
    opra_ct = np.zeros(parameter["fjsp"]["job"]["size"], dtype=int)  # 紀錄每一工件的製程數
    job_with_opra = list()  # 帶有(工件編號, 製程編號)的投料順序

    # 取得原始 OS
    OS = particle[: para_fjsp["opra"]["size"]]

    # 逐個將 os 附上相應的 opra
    for job in OS:
        job = int(job)  # to int
        job_with_opra.append((job, opra_ct[job]))  # (工件編號, 製程編號)
        opra_ct[job] += 1  # 更新
    return job_with_opra


# 將 MS 解譯為帶有(工件編號, 製程編號)的被選中機台
def decoding_MS(particle: np.array, parameter: dict) -> dict:
    # 複製
    para_fjsp = parameter["fjsp"]

    # 初始化
    ct = 0
    job_with_opra = dict()  # 帶有(工件編號, 製程編號)的被選中機台

    # 取得原始 MS
    MS = particle[para_fjsp["opra"]["size"] :]

    # 逐個將 ms 附上相應的 (job, opra)
    for idx, job in enumerate(para_fjsp["job"]["seq"]):
        for opra_size in range(para_fjsp["opra"]["seq"][idx]):
            job_with_opra[(job, opra_size)] = int(MS[ct])
            ct += 1
    return job_with_opra


# 取得初始總族的 gbest 和 pbest
def get_init_best(pop: np.array, fitness: np.array) -> tuple:
    # gbest
    f_min_idx = fitness.argmin()  # 編號
    gbest_f = fitness[f_min_idx]  # 適應值
    gbest_x = pop[f_min_idx].copy()  # 粒子

    # pbest
    pbest_x = pop.copy()
    pbest_f = fitness.copy()
    return (gbest_x, gbest_f, pbest_x, pbest_f)


if __name__ == "__main__":
    # 參數表
    parameter = {
        # 路徑
        "path": {
            "rawdata": "rawdata",
            "result": "result",
            "filename": "data_first.xlsx",
        },
        # 演算法參數
        "alg": {"maxiter": 500, "popsize": 50, "w": 0.9, "c": (2, 2)},
        # 資料
        "data": {"raw": pd.DataFrame, "regular": dict, "normal": dict},
    }

    # 建立絕對路徑
    parameter["path"] = get_abs_path(parameter=parameter)

    # 檢查資料夾是否存在
    check_dir_exist(parameter=parameter)

    # 讀取資料
    parameter["data"]["raw"] = pd.read_excel(
        parameter["path"]["rawdata"], index_col=[0, 1]
    )

    # 移除不可使用的機台
    parameter["data"]["regular"] = regular_data(parameter=parameter)

    # 將工時歸一化後進行排序，最後取累積和
    parameter["data"]["normal"] = normalize_data(parameter=parameter)

    # 取得排程參數
    parameter["fjsp"] = get_fjsp_parameter(parameter=parameter)

    # 初始化種族、速度和計算適應值
    (pop, vel, fitness) = initial_population(parameter=parameter)

    # 取得初始總族的 gbest 和 pbest
    gbest_pop, gbest_fitness, pbest_pop, pbest_fitness = get_init_best(
        pop=pop, fitness=fitness
    )

    # 每代最小的 pbest
    iter_process = np.zeros(parameter["alg"]["maxiter"])
    # 每代的 gbest
    pso_base = np.zeros(parameter["alg"]["maxiter"])

    # 開始計算時間
    begin = time.time()

    # 開始迭代
    for iter_ in range(parameter["alg"]["maxiter"]):
        # 速度更新
        for idx in range(parameter["alg"]["popsize"]):
            vel[idx] = (
                parameter["alg"]["w"] * vel[idx]
                + parameter["alg"]["c"][0]
                * np.random.rand()
                * (pbest_pop[idx] - pop[idx])
                + parameter["alg"]["c"][1] * np.random.rand() * (gbest_pop - pop[idx])
            )

        # 位置更新(工序部分)
        # 當 x' = x + v 以後
        # os 部分會變成範圍未知的浮點數
        # 這時將原先的 os 與 x'(os部分) 綁定在一起
        # 然後按 x'(os部分) 排序，就能得到新的 os
        for idx in range(parameter["alg"]["popsize"]):
            os_before = pop[idx][: parameter["fjsp"]["opra"]["size"]].copy()
            pop[idx] += vel[idx]
            os_after = vel[idx][: parameter["fjsp"]["opra"]["size"]].copy()
            pair = [
                (os_after[i], os_before[i])
                for i in range(parameter["fjsp"]["opra"]["size"])
            ]
            pair = sorted(pair, key=itemgetter(0))
            for i, (_, job) in enumerate(pair):
                pop[idx][i] = job
        pop = np.ceil(pop)

        # 位置更新(機器部分)
        for idx in range(parameter["alg"]["popsize"]):
            ms = decoding_MS(particle=pop[idx], parameter=parameter)
            ct = parameter["fjsp"]["opra"]["size"]
            for (job, opra), machine in ms.items():
                # 如果機台編號超出範圍，或者選到的機台所對應工時為空，則修正為最長工時的機台
                if (
                    machine not in parameter["fjsp"]["machine"]["seq"]
                    or machine not in parameter["data"]["regular"][job][opra].keys()
                ):
                    min_len_machine = list(
                        parameter["data"]["normal"][job][opra].keys()
                    )[-1]
                    pop[idx][ct] = min_len_machine
                ct += 1

        # 紀錄當代最小的 pbest
        iter_process[iter_] = fitness.min()
        # 紀錄當代的 gbest
        pso_base[iter_] = gbest_fitness

        # 計算粒子的適應值(總完工時間)
        for idx in range(parameter["alg"]["popsize"]):
            fitness[idx] = calculate(particle=pop[idx], parameter=parameter)

        # 更新每一粒子的 pbest
        for idx in range(parameter["alg"]["popsize"]):
            if fitness[idx] < pbest_fitness[idx]:
                pbest_fitness[idx] = fitness[idx]
                pbest_pop[idx] = pop[idx].copy()

        # 更新 gbest
        if pbest_fitness.min() < gbest_fitness:
            gbest_fitness = pbest_fitness.min()
            gbest_pop = pop[pbest_fitness.argmin()].copy()

    # 結束計算時間
    end = time.time()

    # 文字打印
    print("按照完全随机初始化的pso算法求得的最好的最大完工时间：", min(pso_base))
    print("按照完全随机初始化的pso算法求得的最好的工艺方案：", gbest_pop)
    print(f"整个迭代过程所耗用的时间：{end - begin:.2f}s")

    # 初始化
    Tm = np.zeros(parameter["fjsp"]["machine"]["size"])  # 每一機台的結束時間
    To = np.zeros(
        (parameter["fjsp"]["job"]["size"], max(parameter["fjsp"]["opra"]["seq"]))
    )  # 每一工件每一製程的結束時間

    OS = decoding_OS(particle=gbest_pop, parameter=parameter)
    MS = decoding_MS(particle=gbest_pop, parameter=parameter)
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
    ax1.plot(pso_base)
    ax2 = plt.subplot(gs[0, 1])
    ax2.set_title("每次迭代后种群适应度最小值的变化情况")
    ax2.plot(iter_process)
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
