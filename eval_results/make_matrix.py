import numpy as np
import pandas as pd

# .npyファイルの読み込み
data = np.load("result_baseball_pitch.npy", allow_pickle=True).item()

# 2重の辞書をDataFrameに変換
# この部分はデータの構造に応じて調整が必要かもしれません
df = pd.DataFrame([{**{"outer_key": k}, **v} for k, v in data.items()])

# CSVファイルとして出力
df.to_csv("output.csv", index=False)
