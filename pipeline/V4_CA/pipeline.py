from typing import overload, Tuple, Optional

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from PIL import Image
from einops import rearrange

from model.V4_CA.cldm import ControlLDM
from model.V4_CA.gaussian_diffusion import Diffusion
from utils.V4_CA.sampler import SpacedSampler
from utils.cond_fn import Guidance
from utils.common import wavelet_decomposition, wavelet_reconstruction, count_vram_usage

from torchvision.utils import save_image

from dataclasses import dataclass
from typing import Optional, Any

# attention capture
import re
from collections import defaultdict

def bicubic_resize(img: np.ndarray, scale: float) -> np.ndarray:
    pil = Image.fromarray(img)
    res = pil.resize(tuple(int(x * scale) for x in pil.size), Image.BICUBIC)
    return np.array(res)


def resize_short_edge_to(imgs: torch.Tensor, size: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h == w:
        new_h, new_w = size, size
    elif h < w:
        new_h, new_w = size, int(w * (size / h))
    else:
        new_h, new_w = int(h * (size / w)), size
    return F.interpolate(imgs, size=(new_h, new_w), mode="bicubic", antialias=True)


def pad_to_multiples_of(imgs: torch.Tensor, multiple: int) -> torch.Tensor:
    _, _, h, w = imgs.size()
    if h % multiple == 0 and w % multiple == 0:
        return imgs.clone()
    # get_pad = lambda x: (x // multiple + 1) * multiple - x
    get_pad = lambda x: (x // multiple + int(x % multiple != 0)) * multiple - x
    ph, pw = get_pad(h), get_pad(w)
    return F.pad(imgs, pad=(0, pw, 0, ph), mode="constant", value=0)


class UltraFusionPipeline:

    def __init__(self, cldm: ControlLDM, diffusion: Diffusion, fidelity_encoder, device: str) -> None:
        self.cldm = cldm
        self.diffusion = diffusion
        self.fidelity_encoder = fidelity_encoder
        self.device = device
        self.final_size: Tuple[int] = None

    def set_final_size(self, lq: torch.Tensor) -> None:
        h, w = lq.shape[2:]
        self.final_size = (h, w)

    @count_vram_usage
    def run(
        self,
        lq2,
        lq1_mscn_norm,
        lq1_color,
        steps: int = 50,
        strength: float = 1.0,
        tiled: bool = False,
        tile_size: int = 512,
        tile_stride: int = 256,
        pos_prompt: str = "",
        neg_prompt: str = "low quality, blurry, low-resolution, noisy, unsharp, weird textures",
        cfg_scale: float = "4.0",
        cond_fn: Guidance = None,
        fidelity_input: torch.Tensor = None,
        consistent_start: torch.Tensor = None,
        # NOTE : For no warping
        args = None,
        #######################
    ) -> torch.Tensor:
        ### preprocess
        lq1_mscn_norm, lq1_color, lq2 = lq1_mscn_norm.cuda(), lq1_color.cuda(), lq2.cuda()
        bs, _, H, W = lq2.shape
        if not tiled:
            assert H == 512 and W == 512, "The image shape must be equal to 512x512"

        # prepare conditon
        lq2 = lq2 * 2 - 1 #[-1, 1]
        if not tiled:
            cond = self.cldm.prepare_condition(lq2, lq1_mscn_norm, lq1_color, pos_prompt)
        else:
            cond, skip_feats = self.cldm.prepare_condition_tiled(lq2, lq1_mscn_norm, lq1_color, pos_prompt, tile_size=tile_size, tile_stride=tile_stride, fidelity_encoder=self.fidelity_encoder, fidelity_input=fidelity_input)

        if args.fcb_type == 'no_use':
            skip_feats = [None] * len(skip_feats)
        
        uncond = None
        old_control_scales = self.cldm.control_scales
        self.cldm.control_scales = [strength] * 13
        x_T = torch.randn((bs, 4, H // 8, W // 8), dtype=torch.float32, device=self.device)
        # lq_latent = self.cldm.vae_encode(lq)
        # noise = torch.randn(lq_latent.shape, dtype=torch.float32, device=self.device)
        # x_T = self.diffusion.q_sample(x_start=lq_latent, t=torch.tensor([999], device=self.device), noise=noise)
        attn_buf, state = None, None
        if args.save_attn:
            state = AttnCaptureState(capture_every=1)
            handles, attn_buf = attach_cross_attn_hooks(self.cldm.controlnet, state)
        
        ### run sampler
        sampler = SpacedSampler(self.diffusion.betas)
        z = sampler.sample(
            model=self.cldm, device=self.device, steps=steps, batch_size=bs, x_size=(4, H // 8, W // 8),
            cond=cond, uncond=uncond, cfg_scale=cfg_scale, x_T=x_T, progress=True,
            progress_leave=True, cond_fn=cond_fn, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride,
            attn_state = state
        )

        if args.save_attn:
            detach_cross_attn_hooks(handles)

        if not tiled:
            sample = self.cldm.vae_decode(z)
        else:
            sample = self.cldm.vae_decode_tiled(z, tile_size // 8, tile_stride // 8, skip_feats, consistent_start)
        
        ### postprocess
        self.cldm.control_scales = old_control_scales
        return sample, attn_buf # [-1 , 1]
    

############################################ NOTE: attn_capture ############################################
@dataclass
class AttnCaptureState:
    step: int = -1                 # 0..(num_steps-1)
    t_value: Optional[float] = None  # scheduler가 넘겨주는 t (tensor이면 int/float로 변환)
    capture_every: int = 1         # 예: 2면 격스텝 캡처
    user_tag: Optional[Any] = None # 필요시 임의 태그 (예: "low/mid/high sigma")

    def begin_step(self, step: int, t):
        self.step = int(step)
        # t가 torch.Tensor일 수 있으므로 안전 변환
        try:
            self.t_value = float(t.item() if hasattr(t, "item") else t)
        except Exception:
            self.t_value = None


def attach_cross_attn_hooks(controlnet, state, name_regex=r"cross_expo_block\.\d+\.attn"):
    """
    Mutual_Attention2D 모듈(=cross_expo_block.*.attn)에 훅을 달아
    각 forward 이후 attention을 수집. state.begin_step(...)로
    매 스텝 업데이트하면, (step, t) 메타가 같이 들어간다.
    """
    buffer = defaultdict(list)
    handles = []
    pat = re.compile(name_regex)

    def make_hook(mod_name):
        def _hook(module, inputs, output):
            if not getattr(module, "save_attn", False):
                return
            # 스텝 다운샘플(간헐 캡처)
            if state.capture_every > 1 and (state.step % state.capture_every) != 0:
                return
            A = getattr(module, "last_attn", None)
            meta = getattr(module, "last_meta", None)
            if A is None or meta is None:
                return
            buffer[mod_name].append({
                'A': A,                # (B,h,Nq,Nk) on CPU
                'meta': meta,          # Hx,Wx,Hy,Wy,heads
                'step': state.step,    # 정수 스텝 인덱스
                't': state.t_value,    # 스케줄러 t 값(실수)
                'name': mod_name,
            })
        return _hook

    for name, m in controlnet.named_modules():
        if pat.search(name):
            if hasattr(m, "save_attn"):
                m.save_attn = True
            handles.append(m.register_forward_hook(make_hook(name)))

    return handles, buffer


def detach_cross_attn_hooks(handles):
    for h in handles:
        h.remove()