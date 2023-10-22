import os

from addict import Dict


# 建立參數表
def set(
    filename: str, maxiter: int, pop_size: int, w: float, c1: float, c2: float
) -> dict:
    # 取得當前路徑
    cwd = os.getcwd()

    parameter = Dict(
        {
            # 路徑
            "path": {
                "filename": filename,  # 檔案名稱
                "rawdata": os.path.join(cwd, "rawdata", filename),  # rawdata 的絕對路徑
                "result": os.path.join(cwd, "result"),  # result 的絕對路徑
            },
            # 演算法參數
            "algorithm": {
                "maxiter": maxiter,
                "popsize": pop_size,
                "w": w,
                "c": (c1, c2),
            },
        }
    )

    # 若檔案不存在就報錯
    if not os.path.exists(parameter["path"]["rawdata"]):
        raise FileNotFoundError(f"{parameter['path']['rawdata']} 不存在!")

    # 若資料夾不存在就自動建立
    if not os.path.exists(parameter["path"]["result"]):
        os.makedirs(parameter["path"]["result"], exist_ok=True)
    return parameter
