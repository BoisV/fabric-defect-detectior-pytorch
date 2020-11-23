import json
import os
from shutil import copyfile

trgt_path = '../data/fabric_data/trgt'
temp_path = '../data/fabric_data/temp'
label_json_path = '../data/fabric_data/label_json'

"""
获取训练所需的数据集的路径及瑕疵类别，并将类别存入txt文件
txt文件行示例
1596034425106_dev001\15926118492155_4755_2_1.json_02
最后两位是瑕疵类别
author：Hongshu Mu
"""


def get_need_image_in_txt():
    label_dirs = os.listdir(label_json_path)
    need_fabric_paths = []
    for dir in label_dirs:
        file_paths = os.listdir(os.path.join(label_json_path, dir))
        for file in file_paths:
            with open(os.path.join(label_json_path, dir, file), mode='r') as f:
                data = json.load(f)

            flaw_type = data['flaw_type']
            if flaw_type in {1, 2, 5, 13}:
                need_fabric_paths.append('%s_%02d' % (
                    os.path.join(dir, file), flaw_type))

    with open('../data/fabric_data/list.txt', mode='w') as target:
        for path in need_fabric_paths:
            target.write(path+'\n')
        target.close()


"""
从数据集中提取中数据
author：Hongshu
"""


def save():
    txt_path = '../data/fabric_data/list.txt'
    image_names = []
    image_defect_class = []
    with open(txt_path, mode='r') as f:
        while f.readable:
            string = f.readline()
            if string == '':
                break
            name = string[:-9]
            cls = string[-3:-1]
            image_names.append(name)
            image_defect_class.append(int(cls))
        f.close()

    for cls, img in zip(image_defect_class, image_names):
        cls_path = os.path.join('../data/DIY_fabric_data/', str(cls))
        cls_trgt_path = os.path.join(cls_path, 'trgt')
        cls_temp_path = os.path.join(cls_path, 'temp')
        cls_label_json_path = os.path.join(cls_path, 'label_json')
        if not os.path.exists(cls_trgt_path):
            os.makedirs(cls_trgt_path)
        if not os.path.exists(cls_temp_path):
            os.makedirs(cls_temp_path)
        if not os.path.exists(cls_label_json_path):
            os.makedirs(cls_label_json_path)

        imgname = img+'.jpg'
        filename1 = os.path.join(trgt_path, imgname)
        filename2 = os.path.join(cls_trgt_path, imgname.split('\\')[-1])
        copyfile(filename1, filename2)

        filename1 = os.path.join(temp_path, imgname)
        filename2 = os.path.join(cls_temp_path, imgname.split('\\')[-1])
        copyfile(filename1, filename2)

        label_json = img + '.json'
        filename1 = os.path.join(label_json_path, label_json)
        filename2 = os.path.join(
            cls_label_json_path, label_json.split('\\')[-1])
        copyfile(filename1, filename2)


if __name__ == "__main__":
    get_need_image_in_txt()
    save()
