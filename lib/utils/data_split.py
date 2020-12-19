import json
import os
from os.path import sep
from shutil import copyfile, rmtree
import numpy as np


def read_images_by_cls(root='../../data/fabric_data', classes=[1, 2, 5, 13], ratio: float = 0.57):
    """
    获取训练所需的数据集的路径及瑕疵类别，并将类别存入txt文件
    txt文件行示例
    1596034425106_dev001\15926118492155_4755_2_1.json_02
    最后两位是瑕疵类别
    author：Hongshu Mu
    """
    label_json_path = os.path.join(root, 'label_json/')

    label_dirs = os.listdir(label_json_path)
    need_fabric_paths = []
    counts = [0 for i in range(0, 30)]
    for dir in label_dirs:
        file_paths = os.listdir(os.path.join(label_json_path, dir))
        for file in file_paths:
            with open(os.path.join(label_json_path, dir, file), mode='r') as f:
                data = json.load(f)

            flaw_type = data['flaw_type']
            counts[flaw_type] += 1
            if flaw_type in classes:
                need_fabric_paths.append(
                    os.path.join(dir, file[:-5]))
    for idx, count in enumerate(counts):
        print((str(idx) + ':' + str(count)), end=' ')
    print()

    idx = np.arange(len(need_fabric_paths))
    np.random.shuffle(idx)
    train_idx = idx[:int(len(need_fabric_paths) * ratio)]
    test_idx = idx[int(len(need_fabric_paths)*ratio):]

    with open(os.path.join(root, 'train.txt'), mode='w') as target:
        for i in train_idx:
            target.write(need_fabric_paths[i]+'\n')
        target.close()
    with open(os.path.join(root, 'test.txt'), mode='w') as target:
        for i in test_idx:
            target.write(need_fabric_paths[i]+'\n')
        target.close()


def save_images(root='../../data/fabric_data', save_path='../../data/DIY_fabric_data/', train=True):
    """
    从数据集中提取中数据
    author：Hongshu Mu
    """
    if train:
        txt_path = os.path.join(root, 'train.txt')
        save_path = os.path.join(save_path, 'train')
    else:
        txt_path = os.path.join(root, 'test.txt')
        save_path = os.path.join(save_path, 'test')
    image_names = []
    with open(txt_path, mode='r') as f:
        while f.readable:
            imgname = f.readline()
            if imgname == '':
                break
            image_names.append(imgname[:-1])
        f.close()

    list_str = []
    trgt_path = os.path.join(save_path, 'trgt')
    if not os.path.exists(trgt_path):
        os.makedirs(trgt_path)
    temp_path = os.path.join(save_path, 'temp')
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)
    label_json_path = os.path.join(save_path, 'label_json')
    if not os.path.exists(label_json_path):
        os.makedirs(label_json_path)

    for idx, img_path in enumerate(image_names):

        if os.path.getsize(os.path.join(root, 'trgt', (img_path+'.jpg'))) == 0 \
                or os.path.getsize(os.path.join(root, 'temp', (img_path+'.jpg'))) == 0 \
                or os.path.getsize(os.path.join(root, 'label_json', (img_path+'.json'))) == 0:
            continue

        # 新bug，有的trgt图很大，结果tmp图很小
        origin = os.path.join(root, 'trgt', (img_path+'.jpg'))
        s1 = os.path.getsize(origin)
        origin2 = os.path.join(root, 'temp', (img_path+'.jpg'))
        s2 = os.path.getsize(origin2)
        avg = (s1+s2)/2.0
        if(min(s1/avg, s2/avg) < 0.5):
            continue

        origin = os.path.join(root, 'trgt', (img_path+'.jpg'))
        trgt_img_name = os.path.join(trgt_path, ('%05d.jpg' % idx))
        copyfile(origin, trgt_img_name)

        origin = os.path.join(root, 'temp', (img_path+'.jpg'))
        temp_img_name = os.path.join(temp_path,  ('%05d.jpg' % idx))
        copyfile(origin, temp_img_name)

        origin = os.path.join(root, 'label_json', (img_path+'.json'))
        label_json_name = os.path.join(label_json_path,  ('%05d.json' % idx))
        copyfile(origin, label_json_name)

        list_str.append(os.path.join('trgt', ('%05d.jpg' % idx))
                        + '&'
                        + os.path.join('temp', ('%05d.jpg' % idx))
                        + '&'
                        + os.path.join('label_json', ('%05d.json' % idx))
                        + '\n')
        with open(os.path.join(save_path, 'list.txt'), mode='w') as f:
            f.writelines(list_str)
            f.close()


def createDIYDataset(root='../../data/fabric_data', save='../../data/DIY_fabric_data/', classes=[1, 2, 5, 13], ratio=4.0/7):
    """
    """
    if os.path.exists(save):
        print(save, "has existed! Now remove it!")
        rmtree(save)
    read_images_by_cls(root=root, classes=classes, ratio=ratio)
    save_images(
        root=root, save_path=save, train=True)
    save_images(
        root=root, save_path=save, train=False)


if __name__ == "__main__":
    createDIYDataset(root='../../data/fabric_data_new')
