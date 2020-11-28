import json
import os
from PIL import Image
from typing import Callable, Optional
import torch
from torchvision.datasets.vision import VisionDataset


class FabricDataset(VisionDataset):
    classes = {0: 'unknown',     1: 'escape_printing',     2: 'clogging_screen',     3: 'broken_hole',     4: 'toe_closing_defects',     5: 'water_stain',     6: 'smudginess',
               7: 'white_stripe',     8: 'hazy_printing',     9: 'billet_defects',     10: 'trachoma',     11: 'color_smear',     12: 'crease',     13: 'false_positive',     14: 'no_alignment'}

    def __init__(self,
                 root: str,
                 train, bool=True,
                 transform: Optional[Callable] = None,
                 target_trainsform: Optional[Callable] = None
                 ) -> None:
        super(FabricDataset, self).__init__(
            root=root, transform=transform, target_transform=target_trainsform)
        list_file = os.path.join(root, 'list.txt')
        self.trgt_images_list = []
        self.temp_images_list = []
        self.json_images_list = []
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
        origin_h, origin_w = trgt_img.height, trgt_img.width
        label = label_json['flaw_type']

        if self.transform is not None:
            trgt_img = self.transform(trgt_img)
            temp_img = self.transform(temp_img)

        h, w = trgt_img.shape[1:]
        x0 = int(label_json['bbox']['x0'] * w / origin_w)
        x1 = int(label_json['bbox']['x1'] * w / origin_w)
        y0 = int(label_json['bbox']['y0'] * h / origin_h)
        y1 = int(label_json['bbox']['y1'] * h / origin_h)
        coordinates = torch.as_tensor([x0, y0, x1-x0, y1-y0], dtype=torch.float32)



        if self.target_transform is not None:
            trgt_img = self.transform(trgt_img)
            temp_img = self.transform(temp_img)

        return trgt_img, label, temp_img, coordinates

    def __len__(self) -> int:
        return len(self.trgt_images_list)
