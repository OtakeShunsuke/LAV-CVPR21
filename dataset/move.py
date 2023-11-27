import os
import shutil
import random
import argparse
import scipy.io
import numpy as np
import sys


def checkinglabel(file, target):
    mat = scipy.io.loadmat(file)
    print(mat["action"][0])
    if mat["action"][0] == target:
        return True
    else:
        return False


# ラベル内容のアウトプット関数
def outputlabel(file):
    mat = scipy.io.loadmat(file)
    return mat["action"][0]


def split_dataset(input_dataset_path, output_dataset_path, split_ratio, target_label):
    # データセット内の動画クリップのリストを取得
    clip_list = os.listdir(os.path.join(input_dataset_path, "frames"))

    # 出力先フォルダを作成
    train_dir = os.path.join(output_dataset_path, "train")
    val_dir = os.path.join(output_dataset_path, "val")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)

    # データセット内の各動画クリップを処理
    for clip in clip_list:
        clip_path = os.path.join(input_dataset_path, "frames", clip)

        # 対応するラベルファイルのパス
        label_file = os.path.join(input_dataset_path, "labels", clip + ".mat")

        label_train_dir = os.path.join(train_dir, outputlabel(label_file))
        label_val_dir = os.path.join(val_dir, outputlabel(label_file))

        os.makedirs(label_train_dir, exist_ok=True)
        os.makedirs(label_val_dir, exist_ok=True)

        if random.random() < split_ratio:
            split_dir = label_train_dir
        else:
            split_dir = label_val_dir

        # 動画クリップの新しいパスを作成
        # ビデオ単位のフォルダ名を取得
        video_folder = os.path.basename(clip_path)
        clip_dest = os.path.join(split_dir, video_folder)

        shutil.copytree(clip_path, clip_dest)

        # # ラベルが"Study"であるかを確認
        # if checkinglabel(label_file, target_label):
        #     # トレーニングとバリデーションセットに分割
        #     if random.random() < split_ratio:
        #         split_dir = train_dir
        #     else:
        #         split_dir = val_dir

        #     # 動画クリップの新しいパスを作成
        #     # ビデオ単位のフォルダ名を取得
        #     video_folder = os.path.basename(clip_path)
        #     clip_dest = os.path.join(split_dir, video_folder)

        #     # 新しいパスに画像をコピー
        #     shutil.copytree(clip_path, clip_dest)

    # 完了メッセージ
    print("データセットの分割が完了しました。")


if __name__ == "__main__":
    # コマンドライン引数のパーサーを作成
    parser = argparse.ArgumentParser(description="データセットをトレーニングとバリデーションに分割するスクリプト")
    parser.add_argument(
        "--split_ratio", type=float, default=0.8, help="トレーニングとバリデーションの割合 (デフォルト: 0.8)"
    )

    # コマンドライン引数を解析
    target_label = "baseball_pitch"

    args = parser.parse_args()
    input_dataset_path = "/root/src/dataset/Penn_Action"
    output_dataset_path = os.path.join(
        "/root/src/dataset/Penn_Action/", "all_divided_dataset"
    )

    # データセットを分割
    split_dataset(
        input_dataset_path, output_dataset_path, args.split_ratio, target_label
    )


# # matファイルの中身を確認するためのコード


# path = sys.argv[1]
# actionDict = dict()
# poseDict = dict()

# files = os.listdir(path)

# for file in files:
#     filePath = os.path.join(path, file)
#     mat = scipy.io.loadmat(filePath)
#     if mat["action"][0] not in actionDict.keys():
#         actionDict[mat["action"][0]] = [filePath]
#     else:
#         actionDict[mat["action"][0]].append(filePath)

#     if mat["pose"][0] not in poseDict.keys():
#         poseDict[mat["pose"][0]] = [filePath]
#     else:
#         poseDict[mat["pose"][0]].append(filePath)

# # 　ファイルパスとアクションの対応をCSVファイルに保存
# with open("action.csv", "w") as f:
#     for key, value in actionDict.items():
#         for v in value:
#             f.write("{},{}\n".format(v, key))
