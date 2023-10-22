import time
from operator import itemgetter

import numpy as np


class PSO:
    def __init__(self, parameter: dict):
        self.popsize = parameter["algorithm"]["popsize"]
        self.maxiter = parameter["algorithm"]["maxiter"]
        self.parameter = parameter

    # 執行
    def run(self):
        # 初始化種族
        pop = self.initial_population()

        # 初始化速度
        vel = np.zeros_like(pop)

        # 計算初始化種族的適應值
        fitness = np.zeros(self.parameter["algorithm"]["popsize"])
        for idx, particle in enumerate(pop):
            fitness[idx] = self.calculate(particle=particle)

        # 取得全域最佳解
        self.gbest = self.get_gbest(population=pop, fitness=fitness)

        # 取得區域最佳解
        pbest = self.get_pbest(population=pop, fitness=fitness)

        # 每代最小的 pbest
        self.iter_process = np.zeros(self.maxiter)

        # 每代的 gbest
        self.pso_base = np.zeros(self.maxiter)

        # 開始計算時間
        begin = time.time()

        # 開始迭代
        for iter_ in range(self.parameter["algorithm"]["maxiter"]):
            for idx in range(self.parameter["algorithm"]["popsize"]):
                # 取得粒子
                particle = pop[idx]
                # 先備份當前 os，待會會用到
                os_before = particle[: self.parameter["fjsp"]["opra"]["size"]].copy()
                # 速度更新
                vel[idx] = (
                    self.parameter["algorithm"]["w"] * vel[idx]
                    + self.parameter["algorithm"]["c"][0]
                    * np.random.rand()
                    * (pbest[idx][1] - particle)
                    + self.parameter["algorithm"]["c"][1]
                    * np.random.rand()
                    * (self.gbest[1] - particle)
                )
                # 位置更新
                particle += vel[idx]
                # 位置更新(工序部分)
                # 當 x' = x + v 以後
                # os 部分會變成範圍未知的浮點數
                # 這時將原先的 os 與 x'(os部分) 綁定在一起
                # 然後按 x'(os部分) 排序，就能得到新的 os
                particle += vel[idx]
                os_after = vel[idx][: self.parameter["fjsp"]["opra"]["size"]].copy()
                pair = [
                    (os_after[i], os_before[i])
                    for i in range(self.parameter["fjsp"]["opra"]["size"])
                ]
                pair = sorted(pair, key=itemgetter(0))
                for i, (_, job) in enumerate(pair):
                    particle[i] = job
                particle[self.parameter["fjsp"]["opra"]["size"] :] = np.ceil(
                    particle[self.parameter["fjsp"]["opra"]["size"] :]
                )

                # 位置更新(機器部分)
                ms = self.decoding_MS(particle=particle)
                ct = self.parameter["fjsp"]["opra"]["size"]
                for (job, opra), machine in ms.items():
                    # 如果機台編號超出範圍，或者選到的機台所對應工時為空，則修正為最長工時的機台
                    if (
                        machine not in self.parameter["fjsp"]["machine"]["seq"]
                        or machine
                        not in self.parameter["data"]["regular"][job][opra].keys()
                    ):
                        min_len_machine = list(
                            self.parameter["data"]["normal"][job][opra].keys()
                        )[-1]
                        particle[ct] = min_len_machine
                    ct += 1

            # 紀錄當代最小的 pbest
            self.iter_process[iter_] = fitness.min()
            # 紀錄當代的 gbest
            self.pso_base[iter_] = self.gbest[0]

            # 計算粒子的適應值(總完工時間)
            for idx in range(self.parameter["algorithm"]["popsize"]):
                fitness[idx] = self.calculate(particle=pop[idx])

            # 更新每一粒子的 pbest
            for idx in range(self.parameter["algorithm"]["popsize"]):
                if fitness[idx] < pbest[idx][0]:
                    pbest[idx] = (fitness[idx], pop[idx].copy())

            # 更新 gbest
            if fitness.min() < self.gbest[0]:
                self.gbest = (fitness.min(), pop[fitness.argmin()].copy())

        # 結束計算時間
        end = time.time()

        # 計算成本
        self.cost = end - begin

    # 初始化種族、速度和計算適應值
    def initial_population(self) -> tuple:
        # 複製
        para_alg = self.parameter["algorithm"]
        para_data = self.parameter["data"]
        para_fjsp = self.parameter["fjsp"]

        # 初始化種族(前半為投料順序，後半為加工機台)
        pop = np.zeros((para_alg["popsize"], para_fjsp["opra"]["size"] * 2))

        # 逐粒子
        for idx in range(para_alg["popsize"]):
            # 取得粒子
            particle = pop[idx]

            # 隨機生成投料順序
            os = list()
            for job, opra_size in zip(
                para_fjsp["job"]["seq"], para_fjsp["opra"]["seq"]
            ):
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
        return pop

    # 計算粒子的適應值(總完工時間)
    def calculate(self, particle: np.array):
        # 複製
        para_fjsp = self.parameter["fjsp"]
        data = self.parameter["data"]["regular"]

        # 初始化
        Tm = np.zeros(para_fjsp["machine"]["size"])  # 每一機台的結束時間
        To = np.zeros(
            (para_fjsp["job"]["size"], max(para_fjsp["opra"]["seq"]))
        )  # 每一工件每一製程的結束時間

        # 投料順序(工件編號, 製程編號)
        OS = self.decoding_OS(particle=particle)
        MS = self.decoding_MS(particle=particle)

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
    def decoding_OS(self, particle: np.array) -> list:
        # 複製
        para_fjsp = self.parameter["fjsp"]

        # 初始化
        opra_ct = np.zeros(
            self.parameter["fjsp"]["job"]["size"], dtype=int
        )  # 紀錄每一工件的製程數
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
    def decoding_MS(self, particle: np.array) -> dict:
        # 複製
        para_fjsp = self.parameter["fjsp"]

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
