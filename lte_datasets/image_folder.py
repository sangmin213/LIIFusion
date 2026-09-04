import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from .datasets import register
from tqdm import tqdm

@register('image-folder')
class ImageFolder(Dataset):

    def __init__(self, root_path, split_file=None, split_key=None, first_k=None,
                 repeat=1, cache='none', oe_path='./load/SICE/OE', ue_path='./load/SICE/UE', motion_path=None):
        self.repeat = repeat
        self.cache = cache

        if split_file is None:
            filenames = sorted(os.listdir(root_path))
        else:
            with open(split_file, 'r') as f:
                filenames = json.load(f)[split_key]
        oe_names = sorted(os.listdir(oe_path))
        ue_names = sorted(os.listdir(ue_path))
        if first_k is not None:
            filenames = filenames[:first_k]
            oe_names = oe_names[:first_k]
            ue_names = ue_names[:first_k]
        if not (len(filenames) == len(oe_names) == len(ue_names)):
            raise ValueError(
                "Label, OE, and UE directories must contain the same number of files"
            )
        if cache not in {'none', 'in_memory'}:
            raise ValueError("cache must be 'none' or 'in_memory'")

        self.files = []
        self.oe_files = []
        self.ue_files = []
        self.to_tensor = transforms.ToTensor()
        for filename, oe_name, ue_name in tqdm(zip(filenames, oe_names, ue_names)):
            file = os.path.join(root_path, filename)
            oe_file = os.path.join(oe_path, oe_name)
            ue_file = os.path.join(ue_path, ue_name)

            if cache == 'none':
                self.files.append(file)
                self.oe_files.append(oe_file)
                self.ue_files.append(ue_file)

            elif cache == 'in_memory':
                self.files.append(self.to_tensor(
                    Image.open(file).convert('RGB')))
                self.oe_files.append(self.to_tensor(
                    Image.open(oe_file).convert('RGB')))
                self.ue_files.append(self.to_tensor(
                    Image.open(ue_file).convert('RGB')))
                
        # Ref: https://github.com/OpenImagingLab/UltraFusion/blob/6d1d37e08788e8751f6e54cb75e2ab099e53818e/dataset/mef_dataset.py
        self.motion_img_dir = motion_path
        if motion_path is not None:
            motion_names = sorted(os.listdir(motion_path))
            self.motion_mask_list = []
            for motion_name in motion_names:
                self.motion_mask_list.append(self.to_tensor(
                    Image.open(os.path.join(motion_path, motion_name)).convert('RGB')))

    def __len__(self):
        return len(self.files) * self.repeat

    def __getitem__(self, idx):
        idx = idx % len(self.files)
        x = self.files[idx]
        
        # -------------------------
        oe = self.oe_files[idx]
        ue = self.ue_files[idx]
        if self.cache == 'none':
            x = self.to_tensor(Image.open(x).convert('RGB'))
            oe = self.to_tensor(Image.open(oe).convert('RGB'))
            ue = self.to_tensor(Image.open(ue).convert('RGB'))
        if x.shape != oe.shape or x.shape != ue.shape:
            raise ValueError(
                "Corresponding Label, OE, and UE images must have equal dimensions"
            )
        
        motion_type = torch.rand(1)
        if self.motion_img_dir is not None and motion_type < 0.5:
            # local motion
            motion_ind = torch.randint(0, len(self.motion_mask_list), (1, )).item()
            mask = self.motion_mask_list[motion_ind]
        else:
            mask = None
        # -------------------------

        if self.cache == 'none':
            return x, oe, ue, mask

        elif self.cache == 'in_memory':
            return x, oe, ue, mask


@register('paired-image-folders')
class PairedImageFolders(Dataset):

    def __init__(self, root_path_1, root_path_2, **kwargs):
        self.dataset_1 = ImageFolder(root_path_1, **kwargs)
        self.dataset_2 = ImageFolder(root_path_2, **kwargs)

    def __len__(self):
        return len(self.dataset_1)

    def __getitem__(self, idx):
        return self.dataset_1[idx], self.dataset_2[idx]
