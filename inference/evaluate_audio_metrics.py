from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from statistics import mean, median
from tempfile import TemporaryDirectory

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from srcV2.data import R2INRDataset, collate_r2inr
from srcV2.inference.infer_video import build_content_unit_model
from srcV2.training.content_unit_common import sanitize_batch
from srcV2.training.common import split_cache_files
from srcV2.utils.audio import _torch_mel_filterbank, load_waveform
from srcV2.utils.common import batch_to_device, get_device


def normalize_waveform(wav: np.ndarray) -> np.ndarray:
    wav = np.asarray(wav, dtype=np.float32).reshape(-1)
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak > 1e-6:
        wav = 0.95 * wav / peak
    return wav.astype(np.float32)


def griffinlim_logmel_to_waveform_array(
    logmel: np.ndarray,
    sample_rate: int,
    n_fft: int,
    hop_length: int,
    win_length: int,
    n_iter: int,
) -> np.ndarray:
    try:
        import librosa

        mel = np.exp(logmel.T).astype(np.float32)
        wav = librosa.feature.inverse.mel_to_audio(
            mel,
            sr=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=win_length,
            power=1.0,
            n_iter=max(1, int(n_iter)),
        )
    except Exception:
        mel_t = torch.from_numpy(np.exp(logmel).astype(np.float32)).transpose(0, 1)
        fb = _torch_mel_filterbank(sample_rate, n_fft, logmel.shape[1], mel_t.dtype, mel_t.device)
        spec_mag = torch.linalg.pinv(fb.float()).matmul(mel_t.float()).clamp_min(1e-8)
        angles = torch.rand_like(spec_mag) * (2.0 * np.pi)
        window = torch.hann_window(win_length, dtype=spec_mag.dtype)
        length = max(1, int(logmel.shape[0] * hop_length))
        complex_spec = torch.polar(spec_mag, angles)
        for _ in range(max(1, int(n_iter))):
            wav_t = torch.istft(
                complex_spec,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                length=length,
            )
            rebuilt = torch.stft(
                wav_t,
                n_fft=n_fft,
                hop_length=hop_length,
                win_length=win_length,
                window=window,
                center=True,
                return_complex=True,
            )
            phase = rebuilt.angle()
            if phase.shape[-1] != spec_mag.shape[-1]:
                phase = phase[:, : spec_mag.shape[-1]]
                if phase.shape[-1] < spec_mag.shape[-1]:
                    phase = torch.nn.functional.pad(phase, (0, spec_mag.shape[-1] - phase.shape[-1]))
            complex_spec = torch.polar(spec_mag, phase)
        wav = wav_t.detach().cpu().numpy()
    return normalize_waveform(wav)


class SpeechBrainHiFiGAN:
    def __init__(self, source: str, savedir: str, device: torch.device, input_scale: str = "logmel"):
        try:
            try:
                from speechbrain.inference.vocoders import HIFIGAN
            except Exception:
                from speechbrain.pretrained import HIFIGAN
        except Exception as exc:
            raise RuntimeError(
                "SpeechBrain HiFi-GAN could not be imported. If speechbrain is already installed, "
                "check that torch and torchaudio are the same compatible version. "
                f"Original error: {type(exc).__name__}: {exc}"
            ) from exc

        run_opts = {"device": str(device)}
        self.model = HIFIGAN.from_hparams(source=source, savedir=savedir, run_opts=run_opts)
        self.device = device
        self.input_scale = input_scale

    def __call__(self, logmel: np.ndarray) -> np.ndarray:
        if self.input_scale == "linear":
            mel_np = np.exp(logmel).astype(np.float32)
        else:
            mel_np = logmel.astype(np.float32)
        mel = torch.from_numpy(mel_np.T).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            wav = self.model.decode_batch(mel)
        return normalize_waveform(wav.detach().float().cpu().numpy())


def build_waveform_synthesizer(args, device: torch.device):
    vocoder = args.vocoder.lower()
    if vocoder == "griffinlim":
        def synthesize(logmel: np.ndarray) -> np.ndarray:
            return griffinlim_logmel_to_waveform_array(
                logmel,
                sample_rate=args.sample_rate,
                n_fft=args.n_fft,
                hop_length=args.hop_length,
                win_length=args.win_length,
                n_iter=args.griffinlim_iters,
            )

        return synthesize
    if vocoder == "speechbrain-hifigan":
        return SpeechBrainHiFiGAN(args.hifigan_source, args.hifigan_savedir, device, args.hifigan_input_scale)
    raise ValueError(f"Unsupported vocoder: {args.vocoder}")


def resolve_source_audio(path: str, data_dir: Path) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    candidates = [
        Path.cwd() / p,
        data_dir.parent / p,
        PROJECT_ROOT / p,
        PROJECT_ROOT / "srcV2" / p,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def reference_segment(
    source_audio: str,
    data_dir: Path,
    start_time: float,
    target_len: int,
    sample_rate: int,
) -> np.ndarray:
    wav, sr = load_waveform(resolve_source_audio(source_audio, data_dir), sample_rate)
    y = wav.squeeze(0).detach().cpu().numpy().astype(np.float32)
    start = max(0, int(round(float(start_time) * int(sr))))
    ref = y[start : start + target_len]
    if ref.size < target_len:
        ref = np.pad(ref, (0, target_len - ref.size))
    return ref.astype(np.float32)


def align_pair(ref: np.ndarray, pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n = min(int(ref.size), int(pred.size))
    if n <= 0:
        return np.zeros(1, dtype=np.float32), np.zeros(1, dtype=np.float32)
    return ref[:n].astype(np.float32), pred[:n].astype(np.float32)


def snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    ref, pred = align_pair(ref, pred)
    noise = ref - pred
    return float(10.0 * np.log10((np.sum(ref * ref) + 1e-12) / (np.sum(noise * noise) + 1e-12)))


def si_snr_db(ref: np.ndarray, pred: np.ndarray) -> float:
    ref, pred = align_pair(ref, pred)
    ref_zm = ref - np.mean(ref)
    pred_zm = pred - np.mean(pred)
    scale = float(np.dot(pred_zm, ref_zm) / (np.dot(ref_zm, ref_zm) + 1e-12))
    target = scale * ref_zm
    noise = pred_zm - target
    return float(10.0 * np.log10((np.sum(target * target) + 1e-12) / (np.sum(noise * noise) + 1e-12)))


def safe_pesq(ref: np.ndarray, pred: np.ndarray, sample_rate: int, mode: str) -> float | None:
    try:
        from pesq import pesq

        ref, pred = align_pair(ref, pred)
        return float(pesq(sample_rate, ref, pred, mode))
    except Exception:
        return None


def safe_estoi(ref: np.ndarray, pred: np.ndarray, sample_rate: int) -> float | None:
    try:
        from pystoi import stoi

        ref, pred = align_pair(ref, pred)
        return float(stoi(ref, pred, sample_rate, extended=True))
    except Exception:
        return None


def safe_visqol(ref: np.ndarray, pred: np.ndarray, sample_rate: int, visqol_bin: str) -> float | None:
    if not visqol_bin:
        return None
    try:
        import re
        import subprocess

        import soundfile as sf

        ref, pred = align_pair(ref, pred)
        with TemporaryDirectory() as tmp:
            ref_path = Path(tmp) / "ref.wav"
            pred_path = Path(tmp) / "pred.wav"
            sf.write(ref_path, ref, sample_rate)
            sf.write(pred_path, pred, sample_rate)
            proc = subprocess.run(
                [
                    visqol_bin,
                    "--reference_file",
                    str(ref_path),
                    "--degraded_file",
                    str(pred_path),
                    "--use_speech_mode",
                ],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
        match = re.search(r"(?:MOS-LQO|MOS|visqol_moslqo)\\D+([0-9]+(?:\\.[0-9]+)?)", proc.stdout)
        return float(match.group(1)) if match else None
    except Exception:
        return None


def safe_utmos(pred: np.ndarray, sample_rate: int, metric) -> float | None:
    if metric is None:
        return None
    try:
        if hasattr(metric, "calculate_wav"):
            wav = torch.from_numpy(pred.astype(np.float32)).view(1, -1)
            scores = metric.calculate_wav(wav, int(sample_rate))
        else:
            scores = metric(pred, rate=sample_rate)
        if isinstance(scores, dict):
            value = next(iter(scores.values()))
        else:
            value = scores
        if isinstance(value, (list, tuple, np.ndarray)):
            value = np.asarray(value).reshape(-1)[0]
        return float(value)
    except Exception:
        return None


def summarize(rows: list[dict], numeric_keys: list[str]) -> dict:
    out = {"count": len(rows)}
    for key in numeric_keys:
        vals = [float(row[key]) for row in rows if row.get(key) not in (None, "", "nan")]
        vals = [v for v in vals if math.isfinite(v)]
        if not vals:
            continue
        out[key] = {
            "count": len(vals),
            "mean": float(mean(vals)),
            "median": float(median(vals)),
            "min": float(min(vals)),
            "max": float(max(vals)),
            "std": float(np.std(vals)),
        }
    return out


def run(args) -> None:
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = get_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    config = dict(ckpt.get("config", {}))
    if config.get("model_type", "content_unit") != "content_unit":
        raise ValueError("This evaluator currently expects a content_unit checkpoint.")
    model = build_content_unit_model(config, device)
    missing, unexpected = model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"[checkpoint] loaded={args.checkpoint} epoch={ckpt.get('epoch')} missing={len(missing)} unexpected={len(unexpected)}")
    model.eval()

    split_seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    split_val_ratio = float(args.val_ratio if args.val_ratio is not None else config.get("val_ratio", 0.1))
    split_limit = args.limit if args.limit is not None else config.get("limit")
    if split_limit is not None:
        split_limit = int(split_limit)

    selected_files = None
    if args.split != "all":
        train_files, val_files = split_cache_files(data_dir, split_val_ratio, split_seed, split_limit)
        selected_files = train_files if args.split == "train" else val_files
        print(
            f"[split] split={args.split} files={len(selected_files)} "
            f"train={len(train_files)} val={len(val_files)} val_ratio={split_val_ratio} seed={split_seed}"
        )
    else:
        print(f"[split] split=all limit={split_limit}")

    ds = R2INRDataset(
        data_dir,
        files=selected_files,
        max_frames=args.max_frames,
        random_crop=False,
        seed=split_seed,
        limit=split_limit if selected_files is None else None,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        collate_fn=collate_r2inr,
    )

    utmos_metric = None
    if args.utmos:
        try:
            import utmos

            utmos_metric = utmos.Score()
        except Exception as exc:
            print(f"[warn] utmos package unavailable: {exc}")
            try:
                import speechmetrics

                utmos_metric = speechmetrics.load("utmos", window=None)
            except Exception as exc2:
                print(f"[warn] speechmetrics UTMOS unavailable: {exc2}")

    rows: list[dict] = []
    metrics = {x.strip().lower() for x in args.metrics.split(",") if x.strip()}
    amp_enabled = device.type == "cuda" and args.amp
    synthesize_waveform = build_waveform_synthesizer(args, device)
    sample_idx = 0
    with torch.inference_mode():
        for batch in tqdm(loader, desc="eval-audio"):
            batch = batch_to_device(batch, device)
            batch = sanitize_batch(batch)
            batch["return_aux"] = True
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(batch)
            pred_mel = outputs["mel"].detach().float().cpu()
            mel_lengths = batch["mel_lengths"].detach().cpu().tolist()
            mel_times = batch["mel_times"].detach().cpu()
            paths = list(batch["paths"])
            source_audios = list(batch.get("source_audios", batch.get("source_videos", [""] * len(paths))))

            for i, path in enumerate(paths):
                mel_len = int(mel_lengths[i])
                logmel = pred_mel[i, :mel_len].numpy().astype(np.float32)
                pred_wav = synthesize_waveform(logmel)
                start_time = float(mel_times[i, 0].item()) if mel_len > 0 else 0.0
                ref_wav = reference_segment(
                    source_audios[i],
                    data_dir=data_dir,
                    start_time=start_time,
                    target_len=int(pred_wav.size),
                    sample_rate=args.sample_rate,
                )
                ref_wav, pred_wav = align_pair(ref_wav, pred_wav)
                row = {
                    "index": sample_idx,
                    "cache_path": path,
                    "source_audio": source_audios[i],
                    "mel_frames": mel_len,
                    "duration_sec": float(len(pred_wav) / args.sample_rate),
                    "ref_rms": float(np.sqrt(np.mean(ref_wav * ref_wav))) if ref_wav.size else 0.0,
                    "pred_rms": float(np.sqrt(np.mean(pred_wav * pred_wav))) if pred_wav.size else 0.0,
                }
                if "snr" in metrics:
                    row["snr_db"] = snr_db(ref_wav, pred_wav)
                    row["si_snr_db"] = si_snr_db(ref_wav, pred_wav)
                if "pesq" in metrics:
                    row["pesq"] = safe_pesq(ref_wav, pred_wav, args.sample_rate, args.pesq_mode)
                if "estoi" in metrics:
                    row["estoi"] = safe_estoi(ref_wav, pred_wav, args.sample_rate)
                if "visqol" in metrics:
                    row["visqol"] = safe_visqol(ref_wav, pred_wav, args.sample_rate, args.visqol_bin)
                if "utmos" in metrics:
                    row["utmos"] = safe_utmos(pred_wav, args.sample_rate, utmos_metric)
                rows.append(row)
                sample_idx += 1

    csv_path = output_dir / "audio_metrics.csv"
    keys = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)

    numeric_keys = ["snr_db", "si_snr_db", "pesq", "estoi", "visqol", "utmos", "ref_rms", "pred_rms"]
    summary = summarize(rows, numeric_keys)
    summary.update(
        {
            "data_dir": str(data_dir),
            "checkpoint": str(args.checkpoint),
            "sample_rate": int(args.sample_rate),
            "vocoder": str(args.vocoder),
            "hifigan_source": str(args.hifigan_source) if args.vocoder == "speechbrain-hifigan" else "",
            "griffinlim_iters": int(args.griffinlim_iters),
            "metrics_requested": sorted(metrics),
            "split": args.split,
            "split_seed": split_seed,
            "split_val_ratio": split_val_ratio,
            "csv": str(csv_path),
        }
    )
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate predicted audio metrics over cached R2INR samples.")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", default="eval_audio_metrics")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--split", default="all", choices=["all", "train", "val"])
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--sample-rate", type=int, default=16000)
    parser.add_argument("--n-fft", type=int, default=1024)
    parser.add_argument("--hop-length", type=int, default=256)
    parser.add_argument("--win-length", type=int, default=1024)
    parser.add_argument("--vocoder", default="griffinlim", choices=["griffinlim", "speechbrain-hifigan"])
    parser.add_argument("--hifigan-source", default="speechbrain/tts-hifigan-ljspeech")
    parser.add_argument("--hifigan-savedir", default="pretrained_models/tts-hifigan-ljspeech")
    parser.add_argument("--hifigan-input-scale", default="logmel", choices=["logmel", "linear"])
    parser.add_argument("--griffinlim-iters", type=int, default=32)
    parser.add_argument("--metrics", default="snr,pesq,estoi", help="Comma list: snr,pesq,estoi,visqol,utmos")
    parser.add_argument("--pesq-mode", default="wb", choices=["wb", "nb"])
    parser.add_argument("--visqol-bin", default="", help="Optional path/name of a VISQOL binary.")
    parser.add_argument("--utmos", action="store_true", help="Try UTMOS through the optional utmos package if installed.")
    parser.add_argument("--amp", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
