from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from srcV2.models import MaskedMelLoss
from srcV2.training.common import model_inputs


def sanitize_batch(batch: dict) -> dict:
    for key in ("video", "landmarks", "mel"):
        if key in batch and torch.is_tensor(batch[key]):
            if not torch.isfinite(batch[key]).all():
                paths = batch.get("paths", [])
                print(f"[warn] non-finite {key}; paths={paths[:4]}")
                batch[key] = torch.nan_to_num(batch[key], nan=0.0, posinf=0.0, neginf=0.0)
    return batch


def make_mismatch_inputs(batch: dict) -> dict | None:
    if batch["video"].shape[0] < 2:
        return None
    out = model_inputs(batch)
    source_keys = {
        "video",
        "landmarks",
        "mouth_valid_mask",
        "video_mask",
        "video_times",
        "video_lengths",
    }
    for key in source_keys:
        if key in out and torch.is_tensor(out[key]):
            out[key] = torch.roll(out[key], shifts=1, dims=0)
    return out


def infer_num_units(files: list[Path]) -> int:
    for path in files:
        item = torch.load(path, map_location="cpu", weights_only=False)
        if "num_speech_units" in item:
            return int(item["num_speech_units"])
        if "speech_units" in item:
            return int(item["speech_units"].max().item()) + 1
    return 0


def model_outputs(model, batch: dict, return_aux: bool = False):
    inputs = model_inputs(batch)
    if return_aux:
        inputs["return_aux"] = True
    out = model(inputs)
    if isinstance(out, dict):
        return out
    return {"mel": out}


def unit_weight_for_epoch(args, epoch: int) -> float:
    if args.unit_loss_weight <= 0:
        return 0.0
    if epoch <= args.unit_warmup_epochs:
        return 0.0
    ramp = max(1, int(args.unit_ramp_epochs))
    progress = min(1.0, max(0.0, (epoch - args.unit_warmup_epochs) / float(ramp)))
    return float(args.unit_loss_weight) * progress


def make_criterion(args, mel_mean, mel_std, device):
    return MaskedMelLoss(
        mel_mean,
        mel_std,
        lambda_mel=args.lambda_mel,
        lambda_delta=args.lambda_delta,
        lambda_delta2=args.lambda_delta2,
        lambda_energy=args.lambda_energy,
        lambda_mfcc=args.lambda_mfcc,
        lambda_flux=args.lambda_flux,
        lambda_voicing=args.lambda_voicing,
        n_mfcc=args.n_mfcc,
        shift_window=args.shift_window,
    ).to(device)


def unit_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    target_mask: torch.Tensor | None = None,
    label_smoothing: float = 0.0,
) -> torch.Tensor:
    logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
    targets = targets.long()
    if targets.shape[1] != logits.shape[1]:
        x = targets.float().unsqueeze(1)
        targets = F.interpolate(x, size=logits.shape[1], mode="nearest").squeeze(1).long()
    if target_mask is not None and target_mask.shape[1] == logits.shape[1]:
        targets = targets.masked_fill(~target_mask.to(targets.device, dtype=torch.bool), -100)
    return F.cross_entropy(
        logits.transpose(1, 2),
        targets,
        ignore_index=-100,
        label_smoothing=float(label_smoothing),
    )
