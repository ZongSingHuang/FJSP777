# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 15:05:23 2019

@author: david
"""

import time

import matplotlib.pyplot as plt
import numpy as np

"""
改变问题规模时需要改动的参数：工件参数workpiece，工序数目process，机器数目machine，
机器选择部分的范围上限，迭代次数（规模小的时候可能降低迭代次数）
"""


# 讀取 RAWDATA
contents = []
with open("data_first.txt") as f:
    string = f.readlines()
    for item in string:
        contents.append(item.strip().split(" "))


# 初始化種族、速度和計算適應值
def init_pop_v_fit():
    # 初始化種族(前半為投料順序，後半為加工機台)
    pop = np.zeros((popsize, total_process * 2))
    # 初始化速度(前半為投料順序，後半為加工機台)
    v = np.zeros((popsize, total_process * 2))
    # 初始化適應值
    fitness = np.zeros(popsize)

    # 逐粒子
    for i in range(popsize):
        # 隨機生成投料順序
        for j in range(workpiece):
            for p in range(process):
                pop[i][j * process + p] = j + 1
        np.random.shuffle(pop[i][:total_process])

        # 隨機生成加工機台
        for j in range(total_process):
            index = np.random.randint(0, machine)
            while contents[j][index] == "-":
                index = np.random.randint(0, machine)
            pop[i][j + total_process] = index + 1

        # 計算粒子的適應值(總完工時間)
        fitness[i] = calculate(pop[i])
    return pop, v, fitness


# 計算粒子的適應值(總完工時間)
def calculate(x):
    # 输入:粒子位置，输出:粒子适应度值
    Tm = np.zeros(machine)  # 每一機台的結束時間
    Te = np.zeros((workpiece, process))  # 每一工件每一製程的結束時間
    array = handle(x)  # 投料順序(工件編號, 製程編號)

    for i in range(total_process):
        # 取得工時
        machine_index = (
            int(x[total_process + (array[i][0] - 1) * process + (array[i][1] - 1)]) - 1
        )  # contents 的 col
        process_index = (array[i][0] - 1) * process + (
            array[i][1] - 1
        )  # contents 的 row
        process_time = int(contents[process_index][machine_index])  # 工時

        # 若為第一道製程，則被選中機台的結束時間直接加上工時
        if array[i][1] == 1:
            Tm[machine_index] += process_time
            Te[array[i][0] - 1][array[i][1] - 1] = Tm[machine_index]
        # 若不為第一道製程，則機台結束時間為 max(前一製程結束時間, 被選中機台結束時間) + 工時
        else:
            Tm[machine_index] = (
                max(Te[array[i][0] - 1][array[i][1] - 2], Tm[machine_index])
                + process_time
            )
            Te[array[i][0] - 1][array[i][1] - 1] = Tm[machine_index]
    return max(Tm)


# 投料順序(工件編號, 製程編號)
def handle(x):
    # 输入：粒子的位置，输出：对工序部分处理后的列表
    piece_mark = np.zeros(workpiece)  # 统计工序的标志
    array = []  # 经过处理后的工序列表
    for i in range(total_process):
        piece_mark[int(x[i] - 1)] += 1
        array.append((int(x[i]), int(piece_mark[int(x[i] - 1)])))
    return array


# 取得初始總族的 gbest 和 pbest
def get_init_best(fitness, pop):
    # 群体最优的粒子位置及其适应度值
    gbestpop, gbestfitness = pop[fitness.argmin()].copy(), fitness.min()
    # 个体最优的粒子位置及其适应度值,使用copy()使得对pop的改变不影响pbestpop，pbestfitness类似
    pbestpop, pbestfitness = pop.copy(), fitness.copy()
    return gbestpop, gbestfitness, pbestpop, pbestfitness


if __name__ == "__main__":
    # 排程參數
    machine = 6  # 機台數目
    workpiece = 10  # 工件數目
    process = 5  # 每一工件的工序數目
    total_process = workpiece * process  # 工件數目 × 工序數目

    # PSO 參數
    maxgen = 500  # 最大迭代次數
    w = 0.9  # 慣性權重
    lr = (2, 2)  # C1, C2
    popsize = 50  # 種族規模
    rangepop = (1, 6)  # 粒子编码中机器选择部分的范围

    # 逐 row 的將 RAWDATA 的工時進行歸一化，並儲存到 clean_contents
    clean_contents = []
    for i in range(total_process):
        # 若工時不為 -，則進行儲存，格式為 [工時, 機台編號]
        clean_contents.append(
            [
                [int(contents[i][j]), j + 1]
                for j in range(machine)
                if contents[i][j] != "-"
            ]
        )
        # 對該 row 的工時取倒數，並且加總為 temp_sum
        temp_sum = 0
        for j in range(len(clean_contents[i])):
            temp_sum += 1 / clean_contents[i][j][0]
        # 對該 row 的工時取倒數，並且除以 temp_sum，進行歸一化
        for j in range(len(clean_contents[i])):
            clean_contents[i][j][0] = (1 / clean_contents[i][j][0]) / temp_sum
        # 按工時，對該 row 由小到大進行排序後，逐一加總
        clean_contents[i].sort()
        cumulation = 0
        for j in range(len(clean_contents[i])):
            cumulation += clean_contents[i][j][0]
            clean_contents[i][j][0] = cumulation

    # 初始化種族、速度和計算適應值
    pop, v, fitness = init_pop_v_fit()
    # 取得初始總族的 gbest 和 pbest
    gbestpop, gbestfitness, pbestpop, pbestfitness = get_init_best(fitness, pop)

    # 每代最小的 pbest
    iter_process = np.zeros(maxgen)
    # 每代的 gbest
    pso_base = np.zeros(maxgen)

    # 開始計算時間
    begin = time.time()

    # 開始迭代
    for i in range(maxgen):
        # 速度更新
        for j in range(popsize):
            v[j] = (
                w * v[j]
                + lr[0] * np.random.rand() * (pbestpop[j] - pop[j])
                + lr[1] * np.random.rand() * (gbestpop - pop[j])
            )

        # 位置更新(工序部分)
        # 說白了就只是 X+V 後，附上原先 X 對應的工件編號，按新的 X 並由小到大排序，得到新的投料順序
        for j in range(popsize):
            store = []
            before = pop[j][:total_process].copy()
            pop[j] += v[j]
            reference = v[j][:total_process].copy()
            for p in range(total_process):
                store.append((reference[p], before[p]))
            store.sort()
            for p in range(total_process):
                pop[j][p] = store[p][1]
        pop = np.ceil(pop)

        # 位置更新(機器部分)
        for j in range(popsize):
            array = handle(pop[j])
            for p in range(total_process):
                # 如果機台編號超出範圍，或者選到的機台所對應工時為空，則修正為最長工時的機台
                if (
                    pop[j][
                        total_process + (array[p][0] - 1) * process + (array[p][1] - 1)
                    ]
                    < rangepop[0]
                    or pop[j][
                        total_process + (array[p][0] - 1) * process + (array[p][1] - 1)
                    ]
                    > rangepop[1]
                ) or (
                    contents[(array[p][0] - 1) * process + (array[p][1] - 1)][
                        int(
                            pop[j][
                                total_process
                                + (array[p][0] - 1) * process
                                + (array[p][1] - 1)
                            ]
                            - 1
                        )
                    ]
                    == "-"
                ):
                    row = (array[p][0] - 1) * process + (array[p][1] - 1)
                    pop[j][
                        total_process + (array[p][0] - 1) * process + (array[p][1] - 1)
                    ] = clean_contents[row][len(clean_contents[row]) - 1][1]

        # 紀錄當代最小的 pbest
        iter_process[i] = fitness.min()
        # 紀錄當代的 gbest
        pso_base[i] = gbestfitness

        # 計算粒子的適應值(總完工時間)
        for j in range(popsize):
            fitness[j] = calculate(pop[j])

        # 更新每一粒子的 pbest
        for j in range(popsize):
            if fitness[j] < pbestfitness[j]:
                pbestfitness[j] = fitness[j]
                pbestpop[j] = pop[j].copy()

        # 更新 gbest
        if pbestfitness.min() < gbestfitness:
            gbestfitness = pbestfitness.min()
            gbestpop = pop[pbestfitness.argmin()].copy()

    # 結束計算時間
    end = time.time()

    # 文字打印
    print("按照完全随机初始化的pso算法求得的最好的最大完工时间：", min(pso_base))
    print("按照完全随机初始化的pso算法求得的最好的工艺方案：", gbestpop)
    print(f"整个迭代过程所耗用的时间：{end - begin:.2f}s")

    # 圖片打印
    plt.rcParams["font.sans-serif"] = ["Microsoft JhengHei"]  # 設定字體，避免中文字亂碼
    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121)
    ax1.set_title("全局最优解的变化情况")
    ax1.plot(pso_base)
    ax2 = fig.add_subplot(122)
    ax2.set_title("每次迭代后种群适应度最小值的变化情况")
    ax2.plot(iter_process)
    plt.show()
