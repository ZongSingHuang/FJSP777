import random
import time
from operator import itemgetter

import numpy as np


class PSO:
    def __init__(self, parameter: dict):
        self.popsize = parameter["algorithm"]["popsize"]
        self.maxiter = parameter["algorithm"]["maxiter"]
        self.w = parameter["algorithm"]["w"]
        self.c1 = parameter["algorithm"]["c"][0]
        self.c2 = parameter["algorithm"]["c"][1]
        self.data = parameter["data"]
        self.job_size = parameter["data"]["job"]["size"]
        self.opra_size = parameter["data"]["opra"]["size"]
        self.machine_size = parameter["data"]["machine"]["size"]

    # 執行
    def run(self):
        # 初始化種族
        population = self.initial_population()

        # 初始化速度
        velocity = np.zeros_like(population)

        # 計算初始化種族的適應值
        fitness = np.zeros(self.popsize)
        for idx, particle in enumerate(population):
            fitness[idx] = self.calc_fitness(particle=particle)

        # 取得全域最佳解
        self.gbest = self.get_gbest(population=population, fitness=fitness)

        # 取得區域最佳解
        pbest = self.get_pbest(population=population, fitness=fitness)

        # 每代最小的 pbest
        self.iter_best_pbest = np.zeros(self.maxiter)

        # 每代的 gbest
        self.iter_gbest = np.zeros(self.maxiter)

        # 開始計算時間
        begin = time.time()

        # 開始迭代
        for iter_ in range(self.maxiter):
            for idx in range(self.popsize):
                # 取得粒子
                particle = population[idx]
                # 先備份當前 os，待會會用到
                os_before = particle[: self.opra_size].copy()
                # 速度更新
                velocity[idx] = (
                    self.w * velocity[idx]
                    + self.c1 * np.random.rand() * (pbest[idx][1] - particle)
                    + self.c2 * np.random.rand() * (self.gbest[1] - particle)
                )
                # 位置更新
                particle += velocity[idx]
                # 修正 os
                # 當 x' = x + v 以後
                # os 部分會變成範圍未知的浮點數
                # 這時將原先的 os 與 x'(os部分) 綁定在一起
                # 然後按 x'(os部分) 排序，就能得到新的 os
                os_after = particle[: self.opra_size].copy()
                pair = [(os_after[i], os_before[i]) for i in range(self.opra_size)]
                pair = sorted(pair, key=itemgetter(0))
                for i, (opra, job) in enumerate(pair):
                    particle[i] = job
                # 修正 ms
                particle[self.opra_size :] = np.ceil(particle[self.opra_size :])
                ms = self.decoding_ms(particle=particle)
                ct = self.opra_size
                for (job, opra), machine in ms.items():
                    # 如果機台編號超出範圍，或者選到的機台所對應工時為空，則修正為最短工時的機台
                    if (
                        machine not in self.data["machine"]["seq"]
                        or machine not in self.data["regular"][(job, opra)].keys()
                    ):
                        min_len_machine = self.data["min_machine_of_opra"][(job, opra)]
                        particle[ct] = random.choice(
                            min_len_machine
                        )  # 別用 np.random.choice，慢到靠北
                    ct += 1

            # 紀錄當代最小的 pbest
            self.iter_best_pbest[iter_] = fitness.min()
            # 紀錄當代的 gbest
            self.iter_gbest[iter_] = self.gbest[0]

            # 計算粒子的適應值(總完工時間)
            for idx, particle in enumerate(population):
                fitness[idx] = self.calc_fitness(particle=particle)

            # 更新每一粒子的 pbest
            for idx in range(self.popsize):
                if fitness[idx] < pbest[idx][0]:
                    pbest[idx] = (fitness[idx], population[idx].copy())

            # 更新 gbest
            if fitness.min() < self.gbest[0]:
                self.gbest = (fitness.min(), population[fitness.argmin()].copy())

        # 結束計算時間
        end = time.time()

        # 計算成本
        self.cost = end - begin

    # 初始化種族
    def initial_population(self) -> np.array:
        # 初始化種族: 前半為投料順序，後半為加工機台
        population = np.zeros((self.popsize, self.opra_size * 2))

        for idx in range(self.popsize):
            # 取得粒子
            particle = population[idx]

            # 建立初始的 os
            os_ = list()
            # 每一 job 有幾個 opra，在 os 終究有幾個 job，然後打亂
            for job, opra_size in self.data["opra"]["seq"].items():
                os_ += [job] * opra_size
            particle[: self.opra_size] = os_
            np.random.shuffle(particle[: self.opra_size])

            # 建立初始的 ms
            ms = list()
            # 從 regular 取得每一 opra 的所有可用機台，並隨機挑選
            for (job, opra), msg in self.data["regular"].items():
                machine = list(msg.keys())
                ms.append(np.random.choice(machine))
            particle[self.opra_size :] = ms
        return population

    # 計算適應值
    def calc_fitness(self, particle: np.array) -> float:
        # 初始化
        Tm = np.zeros(self.machine_size)  # 每一機台的結束時間
        To = np.zeros(
            (self.job_size, max(self.data["opra"]["seq"].values()))
        )  # 每一工件每一製程的結束時間
        Jm = {
            machine: None for machine in self.data["machine"]["seq"]
        }  # 每一機台最後加工的 job(換線用)

        # 投料順序(工件編號, 製程編號)
        os_ = self.decoding_os(particle=particle)
        ms = self.decoding_ms(particle=particle)

        for job, opra in os_:
            # 取得工時
            machine = ms[(job, opra)]  # 機台編號
            length = self.data["regular"][(job, opra)][machine]  # 工時

            # 機台結束時間為 max(前一製程結束時間, 被選中機台結束時間+換線時間) + 工時
            setuptime = self.get_setuptime(
                job=(Jm[machine], job), machine=machine, opra_0_need_setup=True
            )
            Tm[machine] = max(To[job, opra - 1], setuptime + Tm[machine]) + length
            To[job, opra] = Tm[machine]
            Jm[machine] = job
        return max(Tm)

    # 將 os 解譯為帶有(工件編號, 製程編號)的投料順序
    def decoding_os(self, particle: np.array) -> list:
        # 初始化
        opra_ct = self.data["first_opra_of_job"].copy()  # 紀錄每一工件的製程數
        output = list()  # 帶有(工件編號, 製程編號)的投料順序

        # 取得原始 os
        os = particle[: self.opra_size]

        # 逐個將 os 附上相應的 opra
        for job in os:
            job = int(job)  # to int
            output.append((job, opra_ct[job]))  # (工件編號, 製程編號)
            opra_ct[job] += 1  # 更新
        return output

    # 將 ms 解譯為帶有(工件編號, 製程編號)的被選中機台
    def decoding_ms(self, particle: np.array) -> dict:
        # 初始化
        ct = 0
        output = dict()  # 帶有(工件編號, 製程編號)的被選中機台

        # 取得原始 MS
        ms = particle[self.opra_size :]

        # 逐個將 ms 附上相應的 (job, opra)
        for job in self.data["job"]["seq"]:
            for opra_size in range(self.data["opra"]["seq"][job]):
                output[(job, opra_size)] = int(ms[ct])
                ct += 1
        return output

    # 比較前後的 job，得到換線時間
    def get_setuptime(
        self, job: tuple, machine: int, opra_0_need_setup: bool = False
    ) -> float:
        if job[0] is None and opra_0_need_setup:
            return 0
        else:
            return 0

    # 取得全域最佳解
    def get_gbest(self, population: np.array, fitness: np.array) -> dict:
        fitness_min_idx = fitness.argmin()  # 編號
        f = fitness[fitness_min_idx]  # 適應值
        x = population[fitness_min_idx].copy()  # 粒子
        gbest = (f, x)
        return gbest

    # 取得區域最佳解
    def get_pbest(self, population: np.array, fitness: np.array) -> list:
        pbest = [(f, x.copy()) for x, f in zip(population, fitness)]
        return pbest
