"""End-to-end LIIFusion inference: UltraFusion coarse fusion then LIIF refinement."""

import argparse
import math
import re
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from tqdm import tqdm

import lte_datasets
import lte_models
from model.raft.raft import RAFT
from model.V4_CA.cldm import ControlLDM
from model.V4_CA.gaussian_diffusion import Diffusion
from pipeline.V4_CA.pipeline import UltraFusionPipeline
from utils.common import instantiate_from_config
from utils.flow import IMF, backward_warp, forward_backward_consistency_check


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class ModelBundle:
    flow: RAFT
    coarse: UltraFusionPipeline
    fine: torch.nn.Module


def pad_to_square(image):
    _, _, height, width = image.shape
    side = max(height, width)
    return F.pad(image, (0, side - width, 0, side - height), mode="constant", value=0)


def pad_to_multiple(image, multiple):
    height, width = image.shape[-2:]
    pad_h = (multiple - height % multiple) % multiple
    pad_w = (multiple - width % multiple) % multiple
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


def pad_for_tiles(image, tile_size, tile_stride):
    height, width = image.shape[-2:]

    def target(size):
        if size <= tile_size:
            return tile_size
        return tile_size + math.ceil((size - tile_size) / tile_stride) * tile_stride

    pad_h = target(height) - height
    pad_w = target(width) - width
    if pad_h == 0 and pad_w == 0:
        return image
    return F.pad(image, (0, pad_w, 0, pad_h), mode="reflect")


def infer_guide_patch_size(model_spec):
    args = model_spec["args"]["guidenet_spec"]["args"]
    if args.get("model_type") != "conv":
        raise ValueError("LIIFusion inference requires a convolutional guide network")
    kernels = args["kernel_size"]
    strides = args["stride_lst"]
    paddings = args["padding_lst"]
    dilations = args.get("dilation_lst", [1] * len(kernels))
    receptive_field = 1
    jump = 1
    for kernel, stride, padding, dilation in zip(kernels, strides, paddings, dilations):
        effective_kernel = dilation * (kernel - 1) + 1
        receptive_field += (effective_kernel - 1 - 2 * padding) * jump
        jump *= stride
    return receptive_field


def load_models(args):
    coarse_config = OmegaConf.load(args.coarse_config)
    cldm: ControlLDM = instantiate_from_config(coarse_config.model.cldm)
    sd_state = torch.load(
        args.sd_checkpoint, map_location="cpu", weights_only=False
    )["state_dict"]
    cldm.load_pretrained_sd(sd_state)
    cldm.load_controlnet_from_ckpt(
        torch.load(args.ultrafusion_checkpoint, map_location="cpu", weights_only=False)
    )
    cldm.eval().cuda()

    fcb_config = OmegaConf.load(args.fcb_config)
    fidelity_encoder = instantiate_from_config(fcb_config.model.fidelity_encoder)
    fidelity_encoder.load_state_dict(
        torch.load(args.fcb_checkpoint, map_location="cpu", weights_only=False),
        strict=True,
    )
    fidelity_encoder.eval().cuda()

    diffusion: Diffusion = instantiate_from_config(coarse_config.model.diffusion)
    diffusion.cuda()
    coarse = UltraFusionPipeline(
        cldm=cldm,
        diffusion=diffusion,
        fidelity_encoder=fidelity_encoder,
        device="cuda",
    )

    flow = RAFT(SimpleNamespace(dropout=0, alternate_corr=False))
    flow_state = torch.load(args.raft_checkpoint, map_location="cpu", weights_only=False)
    if "state_dict" in flow_state:
        flow_state = flow_state["state_dict"]
    flow_state = OrderedDict(
        (key.removeprefix("module."), value) for key, value in flow_state.items()
    )
    flow.load_state_dict(flow_state, strict=True)
    flow.eval().cuda()

    fine_checkpoint = torch.load(
        args.liif_checkpoint, map_location="cpu", weights_only=False
    )
    fine_spec = fine_checkpoint["model"]
    inferred_patch_size = infer_guide_patch_size(fine_spec)
    checkpoint_config = fine_checkpoint.get("fine_config", {})
    saved_patch_size = checkpoint_config.get("guide_patch_size")
    if saved_patch_size is not None and saved_patch_size != inferred_patch_size:
        raise ValueError(
            "Fine checkpoint guide patch metadata does not match its model definition"
        )
    if args.guide_patch_size is None:
        args.guide_patch_size = inferred_patch_size
    elif args.guide_patch_size != inferred_patch_size:
        raise ValueError(
            f"--guide-patch-size={args.guide_patch_size} does not match the "
            f"LIIF checkpoint receptive field ({inferred_patch_size})"
        )
    saved_guide_type = checkpoint_config.get("guide_type")
    if args.guide_type is None:
        args.guide_type = saved_guide_type or "rgb"
    elif saved_guide_type is not None and args.guide_type != saved_guide_type:
        raise ValueError(
            f"--guide-type={args.guide_type} does not match the fine checkpoint "
            f"({saved_guide_type})"
        )
    saved_scale_max = checkpoint_config.get("scale_max")
    if saved_scale_max is not None and args.scale_max != saved_scale_max:
        raise ValueError(
            f"--scale-max={args.scale_max} does not match the fine checkpoint "
            f"({saved_scale_max})"
        )
    fine = lte_models.make(fine_spec, load_sd=True).eval().cuda()
    return ModelBundle(flow=flow, coarse=coarse, fine=fine)


@torch.no_grad()
def align_exposures(under, over, flow_under, flow_model, coarse_size, raft_iterations):
    side = under.shape[-1]
    flow_under = F.interpolate(
        flow_under,
        size=(coarse_size, coarse_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    flow_over = F.interpolate(
        over,
        size=(coarse_size, coarse_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )

    _, under_to_over = flow_model(
        flow_under * 2 - 1,
        flow_over * 2 - 1,
        iters=raft_iterations,
        test_mode=True,
    )
    _, over_to_under = flow_model(
        flow_over * 2 - 1,
        flow_under * 2 - 1,
        iters=raft_iterations,
        test_mode=True,
    )

    scale = side / coarse_size
    under_to_over = F.interpolate(
        under_to_over, size=(side, side), mode="bicubic", align_corners=False
    ) * scale
    over_to_under = F.interpolate(
        over_to_under, size=(side, side), mode="bicubic", align_corners=False
    ) * scale

    padded_under = pad_to_multiple(under, 16)
    padded_over = pad_to_multiple(over, 16)
    padded_forward = pad_to_multiple(over_to_under, 16)
    padded_backward = pad_to_multiple(under_to_over, 16)
    aligned_under = backward_warp(padded_under, padded_forward)
    _, occlusion = forward_backward_consistency_check(
        padded_backward, padded_forward
    )
    aligned_under = aligned_under * (1.0 - occlusion.unsqueeze(1))
    return aligned_under[..., :side, :side], padded_over[..., :side, :side]


@torch.no_grad()
def coarse_fusion(aligned_under, over, pipeline, args):
    original_size = args.coarse_size
    low_under = F.interpolate(
        aligned_under,
        size=(original_size, original_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    low_over = F.interpolate(
        over,
        size=(original_size, original_size),
        mode="bicubic",
        align_corners=False,
        antialias=True,
    )
    low_under = pad_for_tiles(low_under, args.tile_size, args.tile_stride)
    low_over = pad_for_tiles(low_over, args.tile_size, args.tile_stride)

    structure, color = lte_datasets.wrappers.get_color_and_struct(
        isrgb=True,
        input_img=low_under,
        ksize=7,
        sigmaX=0,
        c=1e-7,
    )
    structure = structure.unsqueeze(0).cuda()
    color = color.unsqueeze(0).cuda()
    fidelity_input = torch.cat([low_over, structure, color], dim=1)
    pipeline_args = SimpleNamespace(fcb_type="org", save_attn=False)
    set_seed(args.seed)
    fused, _ = pipeline.run(
        lq2=low_over,
        lq1_mscn_norm=structure,
        lq1_color=color,
        tiled=True,
        tile_size=args.tile_size,
        tile_stride=args.tile_stride,
        fidelity_input=fidelity_input,
        args=pipeline_args,
    )
    return fused.add(1).div(2).clamp(0, 1)[..., :original_size, :original_size]


def guide_image(image, guide_type):
    image = image.squeeze(0).cpu()
    if guide_type == "rgb":
        return image
    structure, _ = lte_datasets.wrappers.get_color_and_struct(
        isrgb=True,
        input_img=image,
        ksize=7,
        sigmaX=0,
        c=1e-7,
    )
    return structure.repeat(3, 1, 1)


def extract_guide_patches(over, under, width, start, end, patch_size):
    indices = torch.arange(start, end)
    ys = indices // width
    xs = indices % width
    offsets = torch.arange(patch_size)
    grid_y, grid_x = torch.meshgrid(offsets, offsets, indexing="ij")
    full_y = ys[:, None, None] + grid_y
    full_x = xs[:, None, None] + grid_x
    over_patches = over[:, full_y, full_x].permute(1, 0, 2, 3)
    under_patches = under[:, full_y, full_x].permute(1, 0, 2, 3)
    return torch.cat([over_patches, under_patches], dim=1).unsqueeze(0)


def coordinate_batch(height, width, start, end, device):
    indices = torch.arange(start, end, device=device)
    ys = torch.div(indices, width, rounding_mode="floor")
    xs = indices % width
    coord_y = -1 + (2 * ys.float() + 1) / height
    coord_x = -1 + (2 * xs.float() + 1) / width
    return torch.stack([coord_y, coord_x], dim=-1).unsqueeze(0)


@torch.no_grad()
def fine_fusion(coarse, aligned_under, over, model, args):
    height = aligned_under.shape[-2]
    width = aligned_under.shape[-1]
    scale = max(height / args.coarse_size / args.scale_max, 1.0)
    normalized_coarse = (coarse - 0.5) / 0.5
    model.gen_feat(normalized_coarse)

    guide_over = guide_image(over, args.guide_type)
    guide_under = guide_image(aligned_under, args.guide_type)
    padding = args.guide_patch_size // 2
    guide_over = F.pad(
        guide_over, (padding, padding, padding, padding), mode="reflect"
    )
    guide_under = F.pad(
        guide_under, (padding, padding, padding, padding), mode="reflect"
    )
    predictions = []
    total_queries = height * width
    for start in tqdm(
        range(0, total_queries, args.query_batch_size),
        desc="LIIF refinement",
        leave=False,
    ):
        end = min(start + args.query_batch_size, total_queries)
        coord = coordinate_batch(height, width, start, end, coarse.device)
        cell = torch.ones_like(coord)
        cell[..., 0] *= 2 / height * scale
        cell[..., 1] *= 2 / width * scale
        patches = extract_guide_patches(
            guide_over,
            guide_under,
            width,
            start,
            end,
            args.guide_patch_size,
        ).to(coarse.device)
        condition = model.guidance_network(patches)
        predictions.append(model.query_rgb(coord, cell, condition).cpu())

    prediction = torch.cat(predictions, dim=1)[0]
    prediction = prediction.mul(0.5).add(0.5).clamp(0, 1)
    return prediction.view(height, width, 3).permute(2, 0, 1)


def load_pair(under_path, over_path):
    to_tensor = ToTensor()
    under = to_tensor(Image.open(under_path).convert("RGB")).unsqueeze(0)
    over = to_tensor(Image.open(over_path).convert("RGB")).unsqueeze(0)
    if under.shape != over.shape:
        raise ValueError(
            f"Exposure dimensions differ: {under_path} {tuple(under.shape[-2:])}, "
            f"{over_path} {tuple(over.shape[-2:])}"
        )
    return under.cuda(), over.cuda()


def find_exposure(scene, marker):
    candidates = [
        path
        for path in scene.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
        and marker in re.split(r"[^a-z0-9]+", path.stem.lower())
    ]
    if len(candidates) != 1:
        raise ValueError(
            f"Expected exactly one '*{marker}*' image in {scene}, found {len(candidates)}"
        )
    return candidates[0]


def input_pairs(args):
    if args.input_dir:
        root = Path(args.input_dir)
        scenes = sorted(path for path in root.iterdir() if path.is_dir())
        if not scenes:
            raise ValueError(f"No scene directories found in {root}")
        for scene in scenes:
            yield scene.name, find_exposure(scene, "ue"), find_exposure(scene, "oe")
    else:
        if not args.underexposed or not args.overexposed:
            raise ValueError(
                "Provide both --underexposed and --overexposed, or use --input-dir"
            )
        yield Path(args.underexposed).stem, Path(args.underexposed), Path(args.overexposed)


def output_path(args, name, is_batch):
    output = Path(args.output)
    if is_batch:
        output.mkdir(parents=True, exist_ok=True)
        return output / f"{name}.png"
    if output.suffix.lower() not in IMAGE_EXTENSIONS:
        raise ValueError("Direct --output must include a supported image extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    return output


@torch.no_grad()
def run_pair(under_path, over_path, models, args):
    under, over = load_pair(under_path, over_path)
    original_height, original_width = under.shape[-2:]
    flow_under = (
        IMF(under, over) if not args.disable_exposure_correction else under.clone()
    )
    under = pad_to_square(under)
    over = pad_to_square(over)
    flow_under = pad_to_square(flow_under)
    aligned_under, square_over = align_exposures(
        under,
        over,
        flow_under,
        models.flow,
        args.coarse_size,
        args.raft_iterations,
    )
    coarse = coarse_fusion(aligned_under, square_over, models.coarse, args)
    result = fine_fusion(
        coarse, aligned_under, square_over, models.fine, args
    )
    return result[:, :original_height, :original_width]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the complete LIIFusion coarse-to-fine MEF pipeline."
    )
    inputs = parser.add_argument_group("inputs")
    inputs.add_argument("--underexposed", help="Underexposed input image.")
    inputs.add_argument("--overexposed", help="Overexposed input image.")
    inputs.add_argument(
        "--input-dir",
        help="Batch root containing one scene directory per UE/OE pair.",
    )
    inputs.add_argument("--output", required=True, help="Output file or batch directory.")

    checkpoints = parser.add_argument_group("checkpoints")
    checkpoints.add_argument(
        "--sd-checkpoint", default="ckpts/v2-1_512-ema-pruned.ckpt"
    )
    checkpoints.add_argument(
        "--ultrafusion-checkpoint", default="ckpts/ultrafusion.pt"
    )
    checkpoints.add_argument("--fcb-checkpoint", default="ckpts/fcb.pt")
    checkpoints.add_argument("--raft-checkpoint", default="ckpts/raft-sintel.pth")
    checkpoints.add_argument("--liif-checkpoint", required=True)

    parser.add_argument(
        "--coarse-config", default="configs/coarse/ultrafusion.yaml"
    )
    parser.add_argument("--fcb-config", default="configs/coarse/fcb.yaml")
    parser.add_argument("--coarse-size", type=int, default=768)
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--tile-stride", type=int, default=256)
    parser.add_argument("--raft-iterations", type=int, default=20)
    parser.add_argument("--scale-max", type=float, default=4.0)
    parser.add_argument("--guide-patch-size", type=int)
    parser.add_argument("--guide-type", choices=["rgb", "mscn"])
    parser.add_argument("--query-batch-size", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument(
        "--disable-exposure-correction",
        action="store_true",
        help="Disable adaptive IMF exposure correction before flow estimation.",
    )
    return parser.parse_args()


def validate_args(args):
    if args.input_dir and (args.underexposed or args.overexposed):
        raise ValueError("--input-dir cannot be combined with direct image inputs")
    if args.input_dir:
        if not Path(args.input_dir).is_dir():
            raise FileNotFoundError(f"Input directory not found: {args.input_dir}")
        if Path(args.output).exists() and not Path(args.output).is_dir():
            raise ValueError("Batch --output must be a directory")
    else:
        if not args.underexposed or not args.overexposed:
            raise ValueError(
                "Provide both --underexposed and --overexposed, or use --input-dir"
            )
        missing_inputs = [
            path
            for path in (args.underexposed, args.overexposed)
            if not Path(path).is_file()
        ]
        if missing_inputs:
            raise FileNotFoundError(
                "Missing input images: " + ", ".join(missing_inputs)
            )
        if Path(args.output).suffix.lower() not in IMAGE_EXTENSIONS:
            raise ValueError("Direct --output must include a supported image extension")
    if args.coarse_size % 8 != 0:
        raise ValueError("--coarse-size must be divisible by 8 for RAFT and diffusion")
    if args.coarse_size < args.tile_size:
        raise ValueError("--coarse-size must be greater than or equal to --tile-size")
    if args.tile_size % 8 != 0 or args.tile_stride % 8 != 0:
        raise ValueError("--tile-size and --tile-stride must be divisible by 8")
    if args.tile_size <= 0 or not 0 < args.tile_stride <= args.tile_size:
        raise ValueError("--tile-stride must be in (0, tile-size]")
    if args.query_batch_size <= 0:
        raise ValueError("--query-batch-size must be positive")
    if args.guide_patch_size is not None and (
        args.guide_patch_size <= 0 or args.guide_patch_size % 2 == 0
    ):
        raise ValueError("--guide-patch-size must be a positive odd integer")
    if not torch.cuda.is_available():
        raise RuntimeError("LIIFusion inference requires an NVIDIA CUDA GPU")
    required_files = [
        args.coarse_config,
        args.fcb_config,
        args.sd_checkpoint,
        args.ultrafusion_checkpoint,
        args.fcb_checkpoint,
        args.raft_checkpoint,
        args.liif_checkpoint,
    ]
    missing = [path for path in required_files if not Path(path).is_file()]
    if missing:
        raise FileNotFoundError("Missing required files: " + ", ".join(missing))


def main():
    args = parse_args()
    validate_args(args)
    pairs = list(input_pairs(args))
    models = load_models(args)
    is_batch = args.input_dir is not None
    for name, under_path, over_path in tqdm(pairs, desc="LIIFusion"):
        result = run_pair(under_path, over_path, models, args)
        destination = output_path(args, name, is_batch)
        ToPILImage()(result).save(destination)


if __name__ == "__main__":
    main()
