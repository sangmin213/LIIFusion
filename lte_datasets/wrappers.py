import functools
import random
import math
from PIL import Image

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision

from lte_datasets import register
from lte_utils import to_pixel_samples
from lte_utils import make_coord

import cv2

import torch.nn.functional as F

def get_color_and_struct(isrgb, input_img: torch.Tensor, ksize, sigmaX, c):  #input an RGB image

    input_img = input_img.squeeze().cpu().numpy().transpose(1, 2, 0)

    if isrgb==True:
        yuv_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2YUV).astype(np.float32)
        y = np.expand_dims(yuv_img[:,:,0], axis=-1).astype(np.float64)
        u = np.expand_dims(yuv_img[:,:,1], axis=-1).astype(np.float32)
        v = np.expand_dims(yuv_img[:,:,2], axis=-1).astype(np.float32)
    else:
        y = input_img.astype(np.float64)
    #mu = gaussian_filter(y, ksize, ksize/6)
    mu = cv2.GaussianBlur(y, (ksize,ksize), sigmaX).astype(np.float64)
    mu_sq = mu * mu
    sigma = np.sqrt(np.absolute(cv2.GaussianBlur(y*y, (ksize,ksize), sigmaX) - mu_sq)).astype(np.float64)
    mu = np.expand_dims(mu, axis=-1)
    sigma = np.expand_dims(sigma, axis=-1)
    dividend = y.astype(np.float64) - mu
    divisor = sigma + c
    struct = dividend / divisor
    struct = struct.astype(np.float32)
    struct_norm = (struct - struct.min()) / (struct.max() - struct.min() + 1e-6)
    struct_norm = torch.from_numpy(struct_norm).permute(2, 0, 1)
    u = torch.from_numpy(u).permute(2, 0, 1)
    v = torch.from_numpy(v).permute(2, 0, 1)
    img_uv = torch.cat([u, v], dim=0)
    return struct_norm, img_uv

def get_mscn_features(image_tensor, kernel_size=7, C=1e-6):
    """
    Calculates MSCN features on the GPU using convolutions.
    image_tensor: Input tensor of shape (B, C, H, W)
    """
    # Ensure it's a float tensor
    img = image_tensor.float()

    # Create a grayscale version if it's RGB
    if img.shape[1] == 3:
        # Using standard luminosity conversion
        img = 0.299 * img[:, 0:1, :, :] + 0.587 * img[:, 1:2, :, :] + 0.114 * img[:, 2:3, :, :]

    # Calculate local mean
    mean_kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size**2)
    mu = torch.nn.functional.conv2d(img, mean_kernel, padding=kernel_size//2, groups=1)

    # Calculate local variance
    mu_sq = mu * mu
    sigma = torch.sqrt(torch.abs(torch.nn.functional.conv2d(img * img, mean_kernel, padding=kernel_size//2, groups=1) - mu_sq) + C)

    # Calculate MSCN
    mscn = (img - mu) / sigma
    return mscn


@register('sr-implicit-paired')
class SRImplicitPaired(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False, sample_q=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }

@register('sr-implicit-paired-fast')
class SRImplicitPairedFast(Dataset):

    def __init__(self, dataset, inp_size=None, augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]

        s = img_hr.shape[-2] // img_lr.shape[-2] # assume int scale
        if self.inp_size is None:
            h_lr, w_lr = img_lr.shape[-2:]
            h_hr = s * h_lr
            w_hr = s * w_lr
            img_hr = img_hr[:, :h_lr * s, :w_lr * s]
            crop_lr, crop_hr = img_lr, img_hr
        else:
            w_lr = self.inp_size
            x0 = random.randint(0, img_lr.shape[-2] - w_lr)
            y0 = random.randint(0, img_lr.shape[-1] - w_lr)
            crop_lr = img_lr[:, x0: x0 + w_lr, y0: y0 + w_lr]
            w_hr = w_lr * s
            x1 = x0 * s
            y1 = y0 * s
            crop_hr = img_hr[:, x1: x1 + w_hr, y1: y1 + w_hr]

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            x0 = random.randint(0, h_hr - h_lr)
            y0 = random.randint(0, w_hr - w_lr)
            
            hr_coord = hr_coord[x0:x0+self.inp_size, y0:y0+self.inp_size, :]
            hr_rgb = crop_hr[:, x0:x0+self.inp_size, y0:y0+self.inp_size]
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
    
    
def resize_fn(img, size):
    return torchvision.transforms.functional.resize(
        img,
        size,
        interpolation=torchvision.transforms.InterpolationMode.BICUBIC,
        antialias=True,
    )


@register('sr-implicit-downsampled')
class SRImplicitDownsampled(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False, sample_q=None, guide_p_type=None, guide_p_size=None):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment
        self.sample_q = sample_q
        self.guide_p_type = guide_p_type
        self.guide_p_size = guide_p_size

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, oe, ue, mask = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)
        
        # ----------------
        kernel_size = self.guide_p_size
        pad_size = kernel_size // 2
        oe_padded = torch.nn.functional.pad(oe, (pad_size, pad_size, pad_size, pad_size), 'reflect')
        ue_padded = torch.nn.functional.pad(ue, (pad_size, pad_size, pad_size, pad_size), 'reflect')
        # ----------------

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            img = img[:, :round(h_lr * s), :round(w_lr * s)] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
            # ----------------
            oe_crop_hr_padded = oe_padded[:, :round(h_lr * s) + 2*pad_size, :round(w_lr * s) + 2*pad_size] # assume round int
            ue_crop_hr_padded = ue_padded[:, :round(h_lr * s) + 2*pad_size, :round(w_lr * s) + 2*pad_size] # assume round int
            # ----------------
        else:
            w_lr = self.inp_size
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)
            # ----------------
            oe_crop_hr_padded = oe_padded[:, x0: x0 + w_hr + 2*pad_size, y0: y0 + w_hr + 2*pad_size]
            ue_crop_hr_padded = ue_padded[:, x0: x0 + w_hr + 2*pad_size, y0: y0 + w_hr + 2*pad_size]
            # ----------------

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)
            # ----------------
            oe_crop_hr_padded = augment(oe_crop_hr_padded)
            ue_crop_hr_padded = augment(ue_crop_hr_padded)
            # ----------------

        hr_coord, hr_rgb = to_pixel_samples(crop_hr.contiguous())
        
        # ----------------
        _, H, W = ue_crop_hr_padded.shape
        if mask is not None:
            mask = resize_fn(mask, (H, W))
            mask = mask[:1, :, :]
            ue_crop_hr_padded = (1. - mask) * ue_crop_hr_padded
        
        if self.guide_p_type == 'mscn':
            oe_struct, oe_color = get_color_and_struct(isrgb=True, input_img=oe_crop_hr_padded, ksize=7, sigmaX=0, c=0.0000001)
            ue_struct, ue_color = get_color_and_struct(isrgb=True, input_img=ue_crop_hr_padded, ksize=7, sigmaX=0, c=0.0000001)
            
            oe_crop_hr_padded = oe_struct.repeat(3,1,1) # mscn value range : 0~1 
            ue_crop_hr_padded = ue_struct.repeat(3,1,1)

        oe_hr_coord, oe_hr_rgb = to_pixel_samples(oe_crop_hr_padded[:,pad_size:-pad_size, pad_size:-pad_size].contiguous()) # coord = make_coord(), rgb = img flatten : [# of pixels, 3]
        ue_hr_coord, ue_hr_rgb = to_pixel_samples(ue_crop_hr_padded[:,pad_size:-pad_size, pad_size:-pad_size].contiguous())
        
        if self.guide_p_type == 'mscn':
            oe_hr_rgb = oe_hr_rgb[:,:1]
            ue_hr_rgb = ue_hr_rgb[:,:1]
            
        # ----------------

        if self.sample_q is not None:
            sample_lst = torch.randperm(len(hr_coord))[:self.sample_q]

            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]
            # ------- point ---------            
            oe_hr_rgb = oe_hr_rgb[sample_lst]
            ue_hr_rgb = ue_hr_rgb[sample_lst]
            # ------- patch ---------             
            kernel_size = self.guide_p_size
            pad_size = kernel_size // 2  # 19 // 2 = 9

            # extract h, w coordinate
            H_hr, W_hr = crop_hr.shape[-2:]
            y_coords = sample_lst // W_hr
            x_coords = sample_lst % W_hr

            oe_patches = self.extract_patches_vectorized(oe_crop_hr_padded, y_coords, x_coords, kernel_size)
            ue_patches = self.extract_patches_vectorized(ue_crop_hr_padded, y_coords, x_coords, kernel_size)
            
            # # extract small patch
            # oe_patches_list = [
            #     oe_crop_hr_padded[:, y:y + kernel_size, x:x + kernel_size] 
            #     for y, x in zip(y_coords, x_coords)
            # ]
            # ue_patches_list = [
            #     ue_crop_hr_padded[:, y:y + kernel_size, x:x + kernel_size] 
            #     for y, x in zip(y_coords, x_coords)
            # ]

            # # 6. 리스트를 하나의 텐서로 결합합니다.
            # # (최종 shape: [2304, C, 19, 19])
            # oe_patches = torch.stack(oe_patches_list, dim=0) 
            # ue_patches = torch.stack(ue_patches_list, dim=0)

            ''' UNFOLD 
            oe_input = oe_crop_hr.unsqueeze(0) # (1, C, H, W)
            ue_input = ue_crop_hr.unsqueeze(0) # (1, C, H, W)

            # 패치 크기 및 스트라이드 설정
            pad_size = 0
            if self.guide_p_size != None:
                pad_size = self.guide_p_size // 2
            kernel_size = 2 * pad_size + 1

            # 모든 가능한 패치를 (1, C * k_h * k_w, N_patches) 형태로 추출
            # N_patches = H_img * W_img
            oe_patches_unfolded = torch.nn.functional.unfold(oe_input, kernel_size=(kernel_size, kernel_size), stride=1, padding=pad_size)
            ue_patches_unfolded = torch.nn.functional.unfold(ue_input, kernel_size=(kernel_size, kernel_size), stride=1, padding=pad_size)

            sample_indices = torch.tensor(sample_lst, dtype=torch.long)

            # 원하는 패치만 선택 (N_patches 차원에서 인덱싱)
            # 결과: (1, C * k_h * k_w, k)
            oe_selected = oe_patches_unfolded[:, :, sample_indices]
            ue_selected = ue_patches_unfolded[:, :, sample_indices]

            # 패치 텐서 재구성 (k, C, H_patch, W_patch) 또는 (k, C*H_patch*W_patch)
            # PyTorch에서는 보통 (B, C, H, W) 형태를 선호
            # 텐서의 shape를 다시 (k, C, H_patch, W_patch)로 변환
            C = oe_crop_hr.size(0)

            # 최종 결과: (k, C, kernel_size, kernel_size)
            oe_patches = oe_selected.squeeze(0).transpose(0, 1).view(-1, C, kernel_size, kernel_size)
            ue_patches = ue_selected.squeeze(0).transpose(0, 1).view(-1, C, kernel_size, kernel_size)
            '''
            # ----------------

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / crop_hr.shape[-2]
        cell[:, 1] *= 2 / crop_hr.shape[-1]

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb,
            'oe': oe_hr_rgb,
            'ue': ue_hr_rgb,
            'oe_p': oe_patches,
            'ue_p': ue_patches,
        }
    
    def extract_patches_vectorized(self, image_tensor, y_coords, x_coords, kernel_size):
        """
        한 번의 연산으로 이미지에서 모든 패치를 추출합니다.

        Args:
            image_tensor (Tensor): (C, H, W) 모양의 원본 이미지 텐서
            y_coords (Tensor): (N,) 모양의 패치 시작 y좌표 텐서 (N=2304)
            x_coords (Tensor): (N,) 모양의 패치 시작 x좌표 텐서 (N=2304)
            kernel_size (int): 패치 크기 (예: 11)

        Returns:
            Tensor: (N, C, kernel_size, kernel_size) 모양의 패치 텐서
        """
        N = y_coords.shape[0]  # N = 2304
        C = image_tensor.shape[0]

        # 1. (kernel_size, kernel_size) 모양의 패치 내부 오프셋 그리드 생성
        # (GPU에서 생성하도록 device 지정)
        delta_y = torch.arange(kernel_size, device=image_tensor.device)
        delta_x = torch.arange(kernel_size, device=image_tensor.device)
        
        # grid_y, grid_x는 (kernel_size, kernel_size) 모양
        grid_y, grid_x = torch.meshgrid(delta_y, delta_x, indexing='ij')

        # 2. 모든 N개의 패치에 대한 절대 좌표 그리드 생성
        # (N, 1, 1) + (kernel_size, kernel_size) -> (N, kernel_size, kernel_size) (Broadcasting)
        full_y_indices = y_coords.view(N, 1, 1) + grid_y
        full_x_indices = x_coords.view(N, 1, 1) + grid_x

        # 3. 고급 인덱싱(Advanced Indexing)을 사용하여 모든 패치 한 번에 추출
        # image_tensor (C, H, W)에서 (C, N, kernel_size, kernel_size) 모양으로 추출
        patches = image_tensor[:, full_y_indices, full_x_indices]

        # 4. (N, C, kernel_size, kernel_size) 모양으로 변경하여 반환
        return patches.permute(1, 0, 2, 3)


@register('sr-implicit-downsampled-fast')
class SRImplicitDownsampledFast(Dataset):

    def __init__(self, dataset, inp_size=None, scale_min=1, scale_max=None,
                 augment=False):
        self.dataset = dataset
        self.inp_size = inp_size
        self.scale_min = scale_min
        if scale_max is None:
            scale_max = scale_min
        self.scale_max = scale_max
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img = self.dataset[idx]
        s = random.uniform(self.scale_min, self.scale_max)

        if self.inp_size is None:
            h_lr = math.floor(img.shape[-2] / s + 1e-9)
            w_lr = math.floor(img.shape[-1] / s + 1e-9)
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            img = img[:, :h_hr, :w_hr] # assume round int
            img_down = resize_fn(img, (h_lr, w_lr))
            crop_lr, crop_hr = img_down, img
        else:
            h_lr = self.inp_size
            w_lr = self.inp_size
            h_hr = round(h_lr * s)
            w_hr = round(w_lr * s)
            x0 = random.randint(0, img.shape[-2] - w_hr)
            y0 = random.randint(0, img.shape[-1] - w_hr)
            crop_hr = img[:, x0: x0 + w_hr, y0: y0 + w_hr]
            crop_lr = resize_fn(crop_hr, w_lr)

        if self.augment:
            hflip = random.random() < 0.5
            vflip = random.random() < 0.5
            dflip = random.random() < 0.5

            def augment(x):
                if hflip:
                    x = x.flip(-2)
                if vflip:
                    x = x.flip(-1)
                if dflip:
                    x = x.transpose(-2, -1)
                return x

            crop_lr = augment(crop_lr)
            crop_hr = augment(crop_hr)

        hr_coord = make_coord([h_hr, w_hr], flatten=False)
        hr_rgb = crop_hr
        
        if self.inp_size is not None:
            idx = torch.tensor(np.random.choice(h_hr*w_hr, h_lr*w_lr, replace=False))
            hr_coord = hr_coord.view(-1, hr_coord.shape[-1])
            hr_coord = hr_coord[idx, :]
            hr_coord = hr_coord.view(h_lr, w_lr, hr_coord.shape[-1])

            hr_rgb = crop_hr.contiguous().view(crop_hr.shape[0], -1)
            hr_rgb = hr_rgb[:, idx]
            hr_rgb = hr_rgb.view(crop_hr.shape[0], h_lr, w_lr)
        
        cell = torch.tensor([2 / crop_hr.shape[-2], 2 / crop_hr.shape[-1]], dtype=torch.float32)

        return {
            'inp': crop_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }    
    
    
@register('sr-implicit-uniform-varied')
class SRImplicitUniformVaried(Dataset):

    def __init__(self, dataset, size_min, size_max=None,
                 augment=False, gt_resize=None, sample_q=None):
        self.dataset = dataset
        self.size_min = size_min
        if size_max is None:
            size_max = size_min
        self.size_max = size_max
        self.augment = augment
        self.gt_resize = gt_resize
        self.sample_q = sample_q

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_lr, img_hr = self.dataset[idx]
        p = idx / (len(self.dataset) - 1)
        w_hr = round(self.size_min + (self.size_max - self.size_min) * p)
        img_hr = resize_fn(img_hr, w_hr)

        if self.augment:
            if random.random() < 0.5:
                img_lr = img_lr.flip(-1)
                img_hr = img_hr.flip(-1)

        if self.gt_resize is not None:
            img_hr = resize_fn(img_hr, self.gt_resize)

        hr_coord, hr_rgb = to_pixel_samples(img_hr)

        if self.sample_q is not None:
            sample_lst = np.random.choice(
                len(hr_coord), self.sample_q, replace=False)
            hr_coord = hr_coord[sample_lst]
            hr_rgb = hr_rgb[sample_lst]

        cell = torch.ones_like(hr_coord)
        cell[:, 0] *= 2 / img_hr.shape[-2]
        cell[:, 1] *= 2 / img_hr.shape[-1]

        return {
            'inp': img_lr,
            'coord': hr_coord,
            'cell': cell,
            'gt': hr_rgb
        }
