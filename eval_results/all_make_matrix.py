import os
import numpy as np
import pandas as pd

# フォルダのパスを指定
folder_path = "all_actions"

# フォルダ内の全ての.npyファイルを処理
data_list = []
for file_name in os.listdir(folder_path):
    if file_name.endswith(".npy"):
        file_path = os.path.join(folder_path, file_name)
        data = np.load(file_path, allow_pickle=True).item()
        data_list.append(data)

# 2重の辞書をDataFrameに変換
df = pd.DataFrame(
    [{**{"outer_key": k}, **v} for data in data_list for k, v in data.items()]
)

# CSVファイルとして出力
df.to_csv("output.csv", index=False)
