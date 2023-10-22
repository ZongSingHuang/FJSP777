import pandas as pd


# 清洗工時直交表後，從中取得排程參數
def table(parameter: dict) -> dict:
    # 複製
    para_path = parameter["path"]

    # 讀取資料
    raw = pd.read_excel(para_path["rawdata"], index_col=[0, 1])

    # 移除不可使用的機台
    regular = rm_null_length(rawdata=raw)

    # 取得每一製程的最小工時所對應的機台
    min_machine_of_opra = get_min_machine_from_each_opra(rawdata=raw)

    # 取得每一工件的第一製程編號
    first_opra_of_each_job = get_first_opra_of_each_job(rawdata=raw)

    # 機台數目
    machine_size = raw.columns.size
    machine_seq = raw.columns.tolist()

    # 工件數目
    job_seq = raw.index.get_level_values(0).unique().tolist()
    job_size = len(job_seq)

    # 製程數目
    opra_size = raw.shape[0]
    opra_seq = raw.groupby(level=0).size().to_dict()

    # 新增
    output = {
        "raw": raw,
        "regular": regular,
        "min_machine_of_opra": min_machine_of_opra,
        "first_opra_of_each_job": first_opra_of_each_job,
        "machine": {"size": machine_size, "seq": machine_seq},
        "job": {"size": job_size, "seq": job_seq},
        "opra": {"size": opra_size, "seq": opra_seq},
    }
    return output


# 移除不可使用的機台
def rm_null_length(rawdata: pd.DataFrame) -> dict:
    # 複製
    raw = rawdata.copy()

    # 初始化
    output = dict()

    # 按 opra 將不可用機台移除
    for (job, opra), row in raw.iterrows():
        # 新增
        output[(job, opra)] = {k: v for k, v in row.items() if v != "-"}
    return output


# 取得每一製程的最小工時所對應的機台
def get_min_machine_from_each_opra(rawdata: pd.DataFrame) -> dict:
    # 複製
    raw = rawdata.copy()

    # 將不可用機台的工時調整為無限大
    raw.replace("-", 48763, inplace=True)

    # 初始化
    output = dict()

    # 按 opra 取得最小工時的機台
    for (job, opra), row in raw.iterrows():
        # 最小工時
        min_length = row.min()

        # 最小工時對應的機台(可能不只一台)
        is_min = row == min_length

        # 新增
        output[(job, opra)] = row[is_min].index.tolist()
    return output


# # 取得每一工件的第一製程編號
def get_first_opra_of_each_job(rawdata: pd.DataFrame) -> dict:
    # 複製
    raw = rawdata.copy()

    # 將不可用機台的工時調整為無限大
    raw.replace("-", 48763, inplace=True)

    # 按 job 分群
    groups = raw.groupby(level=0)

    # 取得每一 job 的第一製程編號
    output = {job: min(group.index.get_level_values(1)) for job, group in groups}
    return output
