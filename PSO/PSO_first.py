import config
import data_clean
import data_plot
from algorithm import PSO

if __name__ == "__main__":
    # 建立參數表
    parameter = config.set(
        filename="data_first.xlsx", maxiter=500, pop_size=50, w=0.9, c1=2, c2=2
    )

    # 清洗工時直交表後，從中取得排程參數
    parameter["data"] = data_clean.table(parameter=parameter)

    pso = PSO(parameter=parameter)
    result = pso.run()

    # 文字打印
    print(f"按照完全随机初始化的pso算法求得的最好的最大完工时间：{pso.gbest[0]}")
    print(f"按照完全随机初始化的pso算法求得的最好的工艺方案：{pso.gbest[1]}")
    print(f"整个迭代过程所耗用的时间：{pso.cost:.2f}s")

    # 影像輸出
    data_plot.draw(pso=pso, parameter=parameter)
