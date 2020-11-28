import json
import os
from shutil import copyfile


def read_images_by_cls(root='../../data/fabric_data', classes=[1, 2, 5, 13]):
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
    for dir in label_dirs:
        file_paths = os.listdir(os.path.join(label_json_path, dir))
        for file in file_paths:
            with open(os.path.join(label_json_path, dir, file), mode='r') as f:
                data = json.load(f)

            flaw_type = data['flaw_type']
            if flaw_type in classes:
                need_fabric_paths.append(
                    os.path.join(dir, file[:-5]))

    with open(os.path.join(root, 'list.txt'), mode='w') as target:
        for path in need_fabric_paths:
            target.write(path+'\n')
        target.close()


def save_images(root='../../data/fabric_data', save_path='../../data/DIY_fabric_data/'):
    """
    从数据集中提取中数据
    author：Hongshu Mu
    """
    txt_path = os.path.join(root, 'list.txt')
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


def create_diydataset(root='../../data/fabric_data', save='../../data/DIY_fabric_data/'):
    """
    """
    read_images_by_cls(root=root)
    save_images(root=root, save_path=save)


if __name__ == "__main__":
    create_diydataset()