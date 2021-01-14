import json
import os
from typing import Callable, Optional

import torchvision
from PIL import Image
from torchvision.datasets.vision import VisionDataset


class FabricDataset(VisionDataset):
    classes = {0: 'unknown',     1: 'escape_printing',     2: 'clogging_screen',     3: 'broken_hole',     4: 'toe_closing_defects',     5: 'water_stain',     6: 'smudginess',
               7: 'white_stripe',     8: 'hazy_printing',     9: 'billet_defects',     10: 'trachoma',     11: 'color_smear',     12: 'crease',     13: 'false_positive',     14: 'no_alignment'}

    def __init__(self,
                 root: str,
                 train: bool = True,
                 transform: Optional[Callable] = None,
                 target_trainsform: Optional[Callable] = None
                 ) -> None:
        super(FabricDataset, self).__init__(
            root=root, transform=transform, target_transform=target_trainsform)
        list_file = os.path.join(root, 'list.txt')
        self.trgt_images_list = []
        self.temp_images_list = []
        self.json_images_list = []
        self.transform = transform
        with open(list_file, mode='r') as f:
            while f.readable():
                line_str = f.readline()
                if line_str == '':
                    break
                splits = line_str[:-1].split('&')
                self.trgt_images_list.append(splits[0])
                self.temp_images_list.append(splits[1])
                self.json_images_list.append(splits[2])
            f.close()

    def __getitem__(self, index):
        trgt_img = Image.open(os.path.join(
            self.root, self.trgt_images_list[index]))
        temp_img = Image.open(os.path.join(
            self.root, self.temp_images_list[index]))
        with open(os.path.join(self.root, self.json_images_list[index]), mode='r') as f:
            label_json = json.load(f)
        label = label_json['flaw_type']

        x0 = int(label_json['bbox']['x0'])
        x1 = int(label_json['bbox']['x1'])
        y0 = int(label_json['bbox']['y0'])
        y1 = int(label_json['bbox']['y1'])

        # xx、xt分别是瑕疵图片和无瑕疵原图的瑕疵区域片段
        xx = trgt_img.crop((x0, y0, x1, y1))
        xt = temp_img.crop((x0, y0, x1, y1))

        if self.transform is not None:
            xx = self.transform(xx)
            xt = self.transform(xt)

        xx = torchvision.transforms.Resize((224, 224))(xx)
        xt = torchvision.transforms.Resize((224, 224))(xt)

        xx = torchvision.transforms.ToTensor()(xx)
        xt = torchvision.transforms.ToTensor()(xt)
        return xx, xt, label

    def __len__(self) -> int:
        return len(self.trgt_images_list)
