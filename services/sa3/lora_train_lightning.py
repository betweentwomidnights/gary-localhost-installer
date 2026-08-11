#!/usr/bin/env python3
"""SA3 LoRA training on the official Lightning DiffusionCondTrainingWrapper.

This replaces the underfit raw-PyTorch loop for the SA3 backend. The raw loop was
a hand reimplementation of the official training_step and drifted from it in ways
that produced the low-frequency "drone" / postprocess_conv inflation even when the
config matched the Spark exactly (see the ratatat-11 A/B). Running the actual
wrapper makes the training step byte-for-byte the Spark's.

We keep gary's own pieces around it: model staging / weight streaming, the
PreEncodedDataset (random-crop intact), the two VRAM optimizations, and the
`-step=N-epoch=M.safetensors` checkpoint naming that register_lora expects.
"""
from __future__ import annotations

import argparse
import json
import os
import struct
import sys
import time
import traceback
from pathlib import Path

SERVICE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SERVICE_DIR))


def _dump_exit_reason(exc):
    """Write an unhandled exception to <log>.exit and re-print it to stderr.

    train_lora_job redirects our stdout/stderr into the job log, so the stderr
    copy is what actually reaches a user-submitted log. The .exit sidecar
    survives even when the pipe is truncated (segfault, child OOM-kill)."""
    log_path = os.environ.get("UNDERFIT_LOG_PATH") or "lora_train_lightning.log"
    try:
        with open(log_path + ".exit", "w") as f:
            f.write(f"lora_train_lightning.py exited with {type(exc).__name__}: {exc}\n\n")
            traceback.print_exception(type(exc), exc, exc.__traceback__, file=f)
    except Exception:
        pass
    try:
        sys.stderr.write(
            f"\n=== lora_train_lightning.py exited with {type(exc).__name__}: {exc} ===\n")
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
        sys.stderr.flush()
    except Exception:
        pass


def _excepthook(exc_type, exc, tb):
    _dump_exit_reason(exc)
    try:
        sys.__excepthook__(exc_type, exc, tb)
    except Exception:
        pass


# Installed BEFORE the heavy imports so a module-level failure (torch/lightning
# ImportError, CUDA init) still produces a diagnosable traceback.
sys.excepthook = _excepthook

import torch
import pytorch_lightning as pl


def _print_env_diagnostics():
    """Log enough accelerator detail to diagnose CUDA and ROCm environments."""
    try:
        print(f"[env] python  {sys.version.split()[0]}  ({sys.executable})", flush=True)
        if torch.version.hip:
            print(f"[env] torch   {torch.__version__}  (ROCm {torch.version.hip})", flush=True)
        else:
            print(f"[env] torch   {torch.__version__}  (CUDA {torch.version.cuda})", flush=True)
        if torch.cuda.is_available():
            device = torch.cuda.get_device_name(0)
            if torch.version.hip:
                print(f"[env] device  {device} (HIP)", flush=True)
            else:
                cap = "".join(map(str, torch.cuda.get_device_capability(0)))
                print(f"[env] device  {device} (sm{cap})", flush=True)
                print(f"[env] archs   {torch.cuda.get_arch_list()}", flush=True)
        print(f"[env] lightning {pl.__version__} | started {time.strftime('%H:%M:%S')}", flush=True)
    except Exception as exc:
        print(f"[env] diagnostics unavailable: {type(exc).__name__}: {exc}", flush=True)

from underfit.backends import get_backend
from underfit.training.lora import save_lora_step, load_lora_resume
from underfit.utils import stream_checkpoint_into_model
from train_memory import free_pretransform, offload_text_conditioner


def _load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class TrainingProgressLog(pl.Callback):
    """Append one 12-byte record per step to loss_by_timestep.bin.

    This is NOT just a diagnostic: the control center derives the live training
    step from this file's size (`read_sa3_training_step` in lib.rs divides the
    byte length by 12). Drop it and the UI progress indicator sticks at 0.
    Format matches underfit's _LossByTimestepLog exactly: struct "Iff" =
    (uint32 step, float32 t, float32 loss), written to cwd (train_lora_job runs
    us with cwd=<run_dir>/demos, which is where lib.rs looks).
    """

    def __init__(self, path: str | None = None):
        self.path = path or os.path.join(os.getcwd(), "loss_by_timestep.bin")
        self._f = None

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        loss = 0.0
        try:
            if isinstance(outputs, dict) and "loss" in outputs:
                loss = float(outputs["loss"].detach())
            elif outputs is not None and hasattr(outputs, "detach"):
                loss = float(outputs.detach())
        except Exception:
            loss = 0.0
        # `t` (mean sampled timestep) isn't exposed at the callback boundary the
        # way it was inline in the raw loop; only the record count is consumed
        # today, so we record 0.0 rather than invent a value.
        try:
            if self._f is None:
                os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
                self._f = open(self.path, "ab")
            self._f.write(struct.pack("Iff", int(trainer.global_step), 0.0, loss))
            if int(trainer.global_step) % 10 == 0:
                self._f.flush()
        except Exception:
            pass  # progress logging must never break training

    def on_train_end(self, trainer, pl_module):
        if self._f is not None:
            try:
                self._f.close()
            finally:
                self._f = None


class SaveLoraCheckpoint(pl.Callback):
    """Save the LoRA adapter as a .safetensors every N steps, named the way
    train_lora_job.training_checkpoints / register_lora expect."""

    def __init__(self, backend, checkpoint_dir: Path, run_label: str,
                 every: int, lora_cfg: dict, base_model: str | None):
        self.backend = backend
        self.checkpoint_dir = Path(checkpoint_dir)
        self.run_label = run_label
        self.every = max(1, int(every))
        self.lora_cfg = lora_cfg
        self.base_model = base_model
        self._last_saved = -1
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save(self, trainer, pl_module):
        step = int(trainer.global_step)
        if step == self._last_saved:
            return
        self._last_saved = step
        epoch = int(trainer.current_epoch)
        out = self.checkpoint_dir / f"{self.run_label}-step={step}-epoch={epoch}.safetensors"
        save_lora_step(
            self.backend, pl_module.diffusion, self.lora_cfg, out,
            step=step, epoch=epoch, base_model=self.base_model,
        )
        print(f"[checkpoint] saved LoRA at step {step} -> {out.name}", flush=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        step = int(trainer.global_step)
        if step > 0 and step % self.every == 0:
            self._save(trainer, pl_module)

    def on_train_end(self, trainer, pl_module):
        # Guarantee a final checkpoint even if max_steps isn't a multiple of `every`.
        self._save(trainer, pl_module)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--config-file", default="")  # accepted for CLI parity; unused
    parser.add_argument("--save-dir", type=Path, required=True)
    parser.add_argument("--model-config", type=Path, required=True)
    parser.add_argument("--dataset-config", type=Path, required=True)
    parser.add_argument("--pretrained-ckpt-path", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--precision", default="bf16-mixed")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    # Resume from an existing LoRA adapter. Matches the official trainer's
    # --lora_checkpoint: it restores the adapter *weights*, not optimizer/step
    # state, so the step counter restarts (full state resume would need Lightning
    # checkpointing). Not surfaced in the UI yet, but kept wired so the capability
    # doesn't get lost.
    parser.add_argument("--lora-ckpt-path", default="")
    args = parser.parse_args()

    _print_env_diagnostics()

    # Pre-warn (and quiet torch's noisy autotune warnings) on pre-Ampere GPUs,
    # same guards the old lora_train.py entry ran at startup.
    from underfit.utils import check_attention_compute_capability, check_attention_backends
    check_attention_compute_capability()
    check_attention_backends()

    from stable_audio_3.training.diffusion import DiffusionCondTrainingWrapper

    backend = get_backend("sa3")
    model_config = _load_json(args.model_config)
    dataset_config = _load_json(args.dataset_config)
    training = model_config.get("training", {}) or {}

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Build + load the base model (mirrors the underfit staging) ---
    print("[startup] Building model …", flush=True)
    model = backend.create_model(model_config)
    print(f"[startup] Streaming base weights from {args.pretrained_ckpt_path} …", flush=True)
    result = stream_checkpoint_into_model(
        model, str(args.pretrained_ckpt_path), device=device, dtype=torch.float16,
    )
    if result is None:
        from underfit.utils import load_ckpt_state_dict
        backend.load_state_into(
            model, load_ckpt_state_dict(str(args.pretrained_ckpt_path)),
            model_type=model_config.get("model_type"),
        )
    model.to(device)

    # --- Wrap with the official trainer (applies LoRA + casts base precision) ---
    lora_config = training.get("lora_config")
    if lora_config is None:
        raise ValueError("model config is missing training.lora_config")
    base_precision = training.get("base_precision")

    lora_state_dict = None
    if args.lora_ckpt_path:
        print(f"[startup] Resuming LoRA weights from {args.lora_ckpt_path} …", flush=True)
        lora_state_dict, _resume_meta = load_lora_resume(backend, args.lora_ckpt_path)

    print("[startup] Building DiffusionCondTrainingWrapper (official) …", flush=True)
    wrapper = DiffusionCondTrainingWrapper(
        model,
        mask_loss_weight=float(training.get("mask_loss_weight", 1.0)),
        mask_padding_attention=True,
        silence_extension_scale_seconds=float(training.get("silence_extension_scale_seconds", 4.0)),
        use_ema=False,
        log_loss_info=False,
        optimizer_configs=training.get("optimizer_configs"),
        pre_encoded=True,
        cfg_dropout_prob=float(training.get("cfg_dropout_prob", 0.1)),
        timestep_sampler=training.get("timestep_sampler", "trunc_logit_normal"),
        timestep_sampler_options=training.get("timestep_sampler_options", {}) or {},
        inpainting_config=training.get("inpainting") or training.get("inpainting_config"),
        use_effective_length_for_schedule=True,
        sample_rate=model_config.get("sample_rate", 44100),
        sample_size=model_config.get("sample_size"),
        lora_config=lora_config,
        lora_state_dict=lora_state_dict,
        ot_coupling=True,
        base_precision=base_precision,
    )

    # --- VRAM optimizations (VAE + T5 offload) ---
    free_pretransform(wrapper.diffusion)

    # --- Dataset / dataloader (random-crop preserved) ---
    print("[startup] Building dataloader …", flush=True)
    sample_size = model_config.get("sample_size")
    sample_rate = model_config.get("sample_rate", 44100)
    audio_channels = model_config.get("audio_channels", 2)
    train_dl = backend.create_dataloader(
        dataset_config, args.batch_size, sample_size, sample_rate,
        audio_channels=audio_channels, num_workers=args.num_workers, shuffle=True,
        pin_memory=False, persistent_workers=False,
    )

    # Offload T5 after the dataloader exists (needs one warm pass over captions).
    offload_text_conditioner(wrapper.diffusion, train_dl, device)

    # --- Checkpoint callback (gary-compatible .safetensors naming) ---
    checkpoint_dir = args.save_dir / args.name / "checkpoints"
    save_cb = SaveLoraCheckpoint(
        backend, checkpoint_dir, args.name, args.checkpoint_every,
        lora_config, model_config.get("base_model"),
    )

    # CSV logger (the Spark's --logger default). Gives us metrics.csv, including
    # train/mse_loss vs train/mse_masked_loss — the Spark's own check that the
    # padding mask is being applied (if masked ~= unmasked, it isn't).
    logger = pl.loggers.CSVLogger(str(args.save_dir / args.name))

    trainer = pl.Trainer(
        devices=1,
        accelerator="gpu",
        strategy="auto",
        precision=args.precision,
        accumulate_grad_batches=1,
        max_steps=int(args.max_steps),
        # None = no clipping, matching the Spark (see train_lora_job.py note).
        gradient_clip_val=(args.gradient_clip_val or None),
        callbacks=[save_cb, TrainingProgressLog()],
        logger=logger,
        default_root_dir=str(args.save_dir),
        enable_checkpointing=False,   # we save LoRA ourselves via SaveLoraCheckpoint
        enable_model_summary=False,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=0,
        log_every_n_steps=1,
    )

    print(f"[startup] Training for {args.max_steps} steps "
          f"(checkpoint every {args.checkpoint_every}) …", flush=True)
    trainer.fit(wrapper, train_dl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
