import os


def rename():
    label_path = 'D:/Workshop/CV/NSPS/车位标注软件/test/'
    img_path = 'D:/Workshop/CV/NSPS/车位标注软件/test/'

    num = 1051
    for filename in os.listdir(label_path):
        if filename.split('.')[-1] != 'json':
            continue
        newname = 'bjtups_' + str(num)
        os.rename(label_path + filename, label_path + newname + '.json')
        num += 1

    num = 1051
    for filename in os.listdir(label_path):
        if filename.split('.')[-1] != 'mat':
            continue
        newname = 'bjtups_' + str(num)
        os.rename(label_path + filename, label_path + newname + '.mat')
        num += 1

    num = 1051
    for filename in os.listdir(img_path):
        if filename.split('.')[-1] != 'jpg':
            continue
        newname = 'bjtups_' + str(num)
        os.rename(img_path + filename, img_path + newname + '.jpg')
        num += 1


if __name__ == '__main__':
    rename()