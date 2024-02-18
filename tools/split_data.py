import os
import random


def main(path):
    random.seed(0)  # 设置随机种子，保证随机结果可复现

    files_path = path + '/img'
    assert os.path.exists(files_path), "path: '{}' does not exist.".format(files_path)

    val_rate = 0.3

    files_name = sorted([file.split(".")[0] for file in os.listdir(files_path)])
    files_num = len(files_name)
    val_index = random.sample(range(0, files_num), k=int(files_num*val_rate))
    train_files = []
    val_files = []
    for index, file_name in enumerate(files_name):
        if index in val_index:
            val_files.append(file_name)
        else:
            train_files.append(file_name)

    try:
        trainval_f = open(path + "/main/trainval.txt", "w")
        train_f = open(path + "/main/train.txt", "w")
        eval_f = open(path + "/main/val.txt", "w")
        trainval_f.write("\n".join((train_files+val_files)))
        train_f.write("\n".join(train_files))
        eval_f.write("\n".join(val_files))
    except FileExistsError as e:
        print(e)
        exit(1)


if __name__ == '__main__':
    path = '../../Datasets/bjtu-ps'
    main(path)
