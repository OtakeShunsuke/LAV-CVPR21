import numpy as np
import os


def main():
    # データが格納されているディレクトリパス
    data_dir = "/root/src/dataset/penn_action_labels"

    # 新しいnpyファイルのパス
    output_file = "/root/src/dataset/penn_action_labels/all_data.npy"

    # データを格納するリスト
    all_data = dict()

    # ディレクトリ内のファイルを走査
    for root, dirs, files in os.walk(data_dir):
        print(f"files: {files}, root: {root}, dirs: {dirs}")
        for file in files:
            # ファイル名が指定の形式である場合のみ処理
            if file.endswith(".npy") and len(file) == 8:
                # ファイルのパスを生成
                file_path = os.path.join(root, file)

                # データを読み込んでリストに追加
                data = dict()
                data["label"] = np.load(file_path)
                all_data[int(file.split(".")[0])] = data

    # リストをnumpy配列に変換して保存
    all_data = np.array(all_data)
    print(all_data)
    np.save(output_file, all_data)


def test():
    load_file = "/root/src/dataset/penn_action_labels/all_data.npy"
    all_data = np.load(load_file, allow_pickle=True).item()

    print("all_data.shape: ", all_data[2214]["label"])


if __name__ == "__main__":
    main()
    test()
