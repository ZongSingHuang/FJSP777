import pandas as pd
from addict import Dict


# 清洗工時直交表後，從中取得排程參數
def table(parameter: dict) -> dict:
    # 複製
    para_path = parameter["path"]

    # 讀取資料
    parameter["data"]["raw"] = pd.read_excel(para_path["rawdata"], index_col=[0, 1])

    # 移除不可使用的機台
    parameter["data"]["regular"] = regular_data(parameter=parameter)

    # 將工時歸一化後進行排序，最後取累積和
    parameter["data"]["normal"] = normalize_data(parameter=parameter)

    # 取得排程參數
    parameter["fjsp"] = get_fjsp_parameter(parameter=parameter)
    return parameter


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
    para_data = parameter["data"]

    # 初始化
    fjsp = Dict()

    # 機台數目
    fjsp["machine"]["size"] = para_data["raw"].columns.size
    fjsp["machine"]["seq"] = para_data["raw"].columns.tolist()

    # 工件數目
    fjsp["job"]["size"] = len(para_data["regular"].keys())
    fjsp["job"]["seq"] = list(para_data["regular"].keys())

    # 製程數目
    fjsp["opra"]["seq"] = [len(v) for v in para_data["regular"].values()]
    fjsp["opra"]["size"] = sum(fjsp["opra"]["seq"])
    return fjsp
