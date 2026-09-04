"""Train the SwinIR-LIIF fine fusion module used by LIIFusion."""

import argparse
import copy
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

import lte_datasets
import lte_models
import lte_utils


def make_loader(spec, training):
    if spec is None:
        return None
    dataset = lte_datasets.make(spec["dataset"])
    dataset = lte_datasets.make(spec["wrapper"], args={"dataset": dataset})
    workers = spec.get("num_workers", 8 if training else 0)
    return DataLoader(
        dataset,
        batch_size=spec["batch_size"],
        shuffle=training,
        num_workers=workers,
        pin_memory=True,
        persistent_workers=workers > 0,
    )


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def normalization_tensors(config, device):
    data_norm = config.get(
        "data_norm",
        {
            "inp": {"sub": [0], "div": [1]},
            "gt": {"sub": [0], "div": [1]},
        },
    )
    inp_sub = torch.tensor(data_norm["inp"]["sub"], device=device).view(1, -1, 1, 1)
    inp_div = torch.tensor(data_norm["inp"]["div"], device=device).view(1, -1, 1, 1)
    gt_sub = torch.tensor(data_norm["gt"]["sub"], device=device).view(1, 1, -1)
    gt_div = torch.tensor(data_norm["gt"]["div"], device=device).view(1, 1, -1)
    return inp_sub, inp_div, gt_sub, gt_div


def forward_batch(model, batch, norm):
    inp_sub, inp_div, gt_sub, gt_div = norm
    inp = (batch["inp"] - inp_sub) / inp_div
    cond = torch.cat([batch["oe_p"], batch["ue_p"]], dim=2)
    pred = model(inp, batch["coord"], batch["cell"], cond)
    gt = (batch["gt"] - gt_sub) / gt_div
    return pred, gt


@torch.no_grad()
def validate(loader, model, norm, device):
    model.eval()
    average = lte_utils.Averager()
    random_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    try:
        for batch in tqdm(loader, leave=False, desc="val"):
            pred, gt = forward_batch(model, move_batch(batch, device), norm)
            average.add(
                lte_utils.calc_psnr(pred, gt, rgb_range=2).item(),
                pred.shape[0],
            )
    finally:
        random.setstate(random_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
    return average.item()


def load_training_state(config, device):
    resume = config.get("resume")
    checkpoint = None
    if resume:
        if not os.path.isfile(resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {resume}")
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        checkpoint_fine_config = checkpoint.get("fine_config")
        if checkpoint_fine_config and checkpoint_fine_config != fine_config(config):
            raise ValueError(
                "Resume checkpoint guide settings do not match the training config"
            )
        model = lte_models.make(checkpoint["model"], load_sd=True).to(device)
        optimizer = lte_utils.make_optimizer(
            model.parameters(), checkpoint["optimizer"], load_sd=True
        )
        config["model"] = copy.deepcopy(checkpoint["model"])
        config["model"].pop("sd")
        config["optimizer"] = copy.deepcopy(checkpoint["optimizer"])
        config["optimizer"].pop("sd")
        epoch_start = checkpoint["epoch"] + 1
    else:
        model = lte_models.make(config["model"]).to(device)
        optimizer = lte_utils.make_optimizer(model.parameters(), config["optimizer"])
        epoch_start = 1

    scheduler = None
    if config.get("multi_step_lr"):
        scheduler = MultiStepLR(optimizer, **config["multi_step_lr"])
        if checkpoint is not None:
            if checkpoint.get("scheduler") is not None:
                scheduler.load_state_dict(checkpoint["scheduler"])
            else:
                scheduler.last_epoch = checkpoint["epoch"]
                scheduler._last_lr = [
                    group["lr"] for group in optimizer.param_groups
                ]
    best_psnr = checkpoint.get("best_psnr", float("-inf")) if checkpoint else float("-inf")
    return model, optimizer, scheduler, epoch_start, best_psnr


def fine_config(config):
    wrapper_args = config["train_dataset"]["wrapper"]["args"]
    return {
        "guide_type": wrapper_args["guide_p_type"],
        "guide_patch_size": wrapper_args["guide_p_size"],
        "scale_max": wrapper_args["scale_max"],
    }


def save_checkpoint(path, model, optimizer, scheduler, config, epoch, best_psnr):
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    model_spec = copy.deepcopy(config["model"])
    model_spec["sd"] = raw_model.state_dict()
    optimizer_spec = copy.deepcopy(config["optimizer"])
    optimizer_spec["sd"] = optimizer.state_dict()
    torch.save(
        {
            "model": model_spec,
            "optimizer": optimizer_spec,
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "fine_config": fine_config(config),
            "best_psnr": best_psnr,
            "epoch": epoch,
        },
        path,
    )


def train(config, output_dir, device, use_amp):
    output_dir.mkdir(parents=True, exist_ok=True)
    log, writer = lte_utils.set_save_path(str(output_dir), remove=False)
    train_loader = make_loader(config["train_dataset"], training=True)
    val_loader = make_loader(config.get("val_dataset"), training=False)
    norm = normalization_tensors(config, device)
    model, optimizer, scheduler, epoch_start, best_psnr = load_training_state(
        config, device
    )
    with (output_dir / "config.yaml").open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    log(f"model: #params={lte_utils.compute_num_params(model, text=True)}")

    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    loss_fn = nn.L1Loss()
    global_step = (epoch_start - 1) * len(train_loader)

    for epoch in range(epoch_start, config["epoch_max"] + 1):
        model.train()
        train_loss = lte_utils.Averager()
        for batch in tqdm(train_loader, leave=False, desc=f"train {epoch}"):
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                pred, gt = forward_batch(model, batch, norm)
                loss = loss_fn(pred, gt)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss.add(loss.item(), pred.shape[0])
            writer.add_scalar("loss/train", loss.item(), global_step)
            global_step += 1

        if scheduler is not None:
            scheduler.step()
        writer.add_scalar("learning_rate", optimizer.param_groups[0]["lr"], epoch)
        message = f"epoch {epoch}/{config['epoch_max']}, loss={train_loss.item():.6f}"
        is_best = False
        if val_loader is not None and config.get("epoch_val") and epoch % config["epoch_val"] == 0:
            raw_model = model.module if isinstance(model, nn.DataParallel) else model
            val_psnr = validate(val_loader, raw_model, norm, device)
            writer.add_scalar("psnr/val", val_psnr, epoch)
            message += f", val_psnr={val_psnr:.4f}"
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                is_best = True

        save_checkpoint(
            output_dir / "epoch-last.pth",
            model,
            optimizer,
            scheduler,
            config,
            epoch,
            best_psnr,
        )
        if config.get("epoch_save") and epoch % config["epoch_save"] == 0:
            save_checkpoint(
                output_dir / f"epoch-{epoch}.pth",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_psnr,
            )
        if is_best:
            save_checkpoint(
                output_dir / "epoch-best.pth",
                model,
                optimizer,
                scheduler,
                config,
                epoch,
                best_psnr,
            )
        log(message)
        writer.flush()

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train LIIFusion's SwinIR-LIIF fine fusion module."
    )
    parser.add_argument(
        "--config",
        default="configs/fine/train_swinir-liif.yaml",
        help="Training YAML configuration.",
    )
    parser.add_argument("--name", default="swinir-liif", help="Run name under --save-dir.")
    parser.add_argument("--save-dir", default="save", help="Checkpoint root directory.")
    parser.add_argument("--gpu", default="0", help="CUDA_VISIBLE_DEVICES value.")
    parser.add_argument("--seed", type=int, default=231)
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision.")
    return parser.parse_args()


def main():
    args = parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if not torch.cuda.is_available():
        raise RuntimeError("LIIFusion training requires an NVIDIA CUDA GPU")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    with open(args.config, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    train(
        config,
        Path(args.save_dir) / args.name,
        torch.device("cuda"),
        use_amp=not args.no_amp,
    )


if __name__ == "__main__":
    main()
