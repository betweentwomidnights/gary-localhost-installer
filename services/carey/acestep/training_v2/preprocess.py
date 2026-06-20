"""
Two-Pass CLI Preprocessing for ACE-Step Training V2.

Converts raw audio files into ``.pt`` tensor files compatible with
``PreprocessedDataModule``.  Uses upstream sub-functions directly and
loads models **sequentially** to minimise peak VRAM:

    Pass 1 (Light ~3 GB):  VAE + Text Encoder  -> intermediate ``.tmp.pt``
    Pass 2 (Heavy ~6 GB):  DIT encoder          -> final ``.pt``

Input modes:
    * With ``--dataset-json``: rich per-sample metadata (lyrics, genre, BPM, …)
    * Without JSON: scan directory, default to ``[Instrumental]``, filename caption
"""

from __future__ import annotations

import logging
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# Split-out helpers
from acestep.training_v2.preprocess_discovery import (
    discover_audio_files as _discover_audio_files,
    load_dataset_metadata as _load_dataset_metadata,
    load_sample_metadata as _load_sample_metadata,
)
from acestep.training_v2.preprocess_prompt import (
    build_simple_prompt as _build_simple_prompt,
)
from acestep.training_v2.preprocess_vae import (
    TARGET_SR as _TARGET_SR,
    tiled_vae_encode as _tiled_vae_encode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_audio_files(
    audio_dir: Optional[str],
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 240.0,
    dataset_json: Optional[str] = None,
    device: str = "auto",
    precision: str = "auto",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
) -> Dict[str, Any]:
    """Preprocess audio files into .pt tensor format (two-pass pipeline).

    Audio files are discovered from one of two sources:

    * **Dataset JSON** (preferred): each entry's ``audio_path`` or
      ``filename`` field locates the audio file directly.
    * **Audio directory** (fallback): scanned **recursively** for
      supported audio formats when no JSON is provided.

    The resulting tensors are adapter-agnostic: they work for both LoRA
    and LoKR training (the adapter type only affects weight injection,
    not the data pipeline).

    Args:
        audio_dir: Directory containing audio files (scanned recursively).
            May be ``None`` when *dataset_json* provides audio paths.
        output_dir: Directory for output .pt files.
        checkpoint_dir: Path to ACE-Step model checkpoints.
        variant: Model variant (turbo, base, sft).
        max_duration: Maximum audio duration in seconds.
        dataset_json: Optional JSON file with per-sample metadata and
            audio paths.
        device: Target device (``"auto"`` to auto-detect).
        precision: Target precision (``"auto"`` to auto-detect).
        progress_callback: ``(current, total, message) -> None``.
        cancel_check: ``() -> bool`` -- return True to cancel.

    Returns:
        Dict with keys: ``processed``, ``failed``, ``total``, ``output_dir``.
    """
    from acestep.training_v2.gpu_utils import detect_gpu

    gpu = detect_gpu(device, precision)
    dev = gpu.device
    prec = gpu.precision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # -- Discover audio files -----------------------------------------------
    audio_files = _discover_audio_files(audio_dir, dataset_json)
    if not audio_files:
        logger.warning("[Side-Step] No audio files found")
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(out_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    # -- Load metadata -------------------------------------------------------
    sample_meta = _load_sample_metadata(dataset_json, audio_files)
    ds_meta = _load_dataset_metadata(dataset_json)

    # Apply dataset-level custom_tag as fallback for samples without one
    ds_tag = ds_meta.get("custom_tag", "")
    if ds_tag:
        for sm in sample_meta.values():
            if not sm.get("custom_tag"):
                sm["custom_tag"] = ds_tag

    # -- Pass 1: VAE + Text Encoder -----------------------------------------
    intermediates, pass1_failed = _pass1_light(
        audio_files=audio_files,
        sample_meta=sample_meta,
        ds_meta=ds_meta,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        max_duration=max_duration,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    # -- Pass 2: DIT Encoder ------------------------------------------------
    processed, pass2_failed = _pass2_heavy(
        intermediates=intermediates,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )
    manifest_entries = _write_variant_manifest(
        out_path=out_path,
        audio_files=audio_files,
        sample_meta=sample_meta,
        ds_meta=ds_meta,
    )

    failed = pass1_failed + pass2_failed
    result = {
        "processed": processed,
        "processed_tensors": processed,
        "failed": failed,
        "total": total,
        "audio_files": total,
        "output_dir": str(out_path),
        "manifest_samples": manifest_entries,
    }
    logger.info(
        "[Gary] Preprocessing complete: %d tensor files from %d audio files, "
        "%d manifest entries, %d failed",
        processed, total, manifest_entries, failed,
    )
    return result


# ---------------------------------------------------------------------------
# Pass 1 -- Light models (VAE + Text Encoder)
# ---------------------------------------------------------------------------

def _pass1_light(
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    ds_meta: Dict[str, Any],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    max_duration: float,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[List[Path], int]:
    """Load audio, VAE-encode, text-encode, save intermediates.

    Args:
        ds_meta: Dataset-level metadata (``tag_position``, ``genre_ratio``,
            ``custom_tag``) from the JSON's top-level ``metadata`` block.

    Returns ``(list_of_intermediate_paths, fail_count)``.
    """
    from acestep.training_v2.model_loader import (
        load_vae,
        load_text_encoder,
        load_silence_latent,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_audio import load_audio_stereo
    from acestep.training.dataset_builder_modules.preprocess_text import encode_text
    from acestep.training.dataset_builder_modules.preprocess_lyrics import encode_lyrics

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 1/2: Loading VAE + Text Encoder ...")
    vae = load_vae(checkpoint_dir, device, precision)
    tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
    silence_latent = load_silence_latent(checkpoint_dir, device, precision, variant=variant)

    intermediates: List[Path] = []
    failed = 0
    total = len(audio_files)

    # Dataset-level prompt settings from ACE-Step's metadata block
    tag_position = ds_meta.get("tag_position", "prepend")
    genre_ratio = ds_meta.get("genre_ratio", 0)
    genre_variant_count = sum(
        1
        for af in audio_files
        if "genre" in _prompt_variants_for_sample(sample_meta.get(af.name, {}), genre_ratio)
    )
    if genre_variant_count:
        logger.info(
            "[Gary] genre_ratio=%d%% -- encoding genre prompts for %d/%d tracks "
            "(training will swap a rotating subset each epoch)",
            genre_ratio, genre_variant_count, total,
        )
    if tag_position != "prepend":
        logger.info("[Gary] tag_position=%s (from dataset metadata)", tag_position)

    try:
        for i, af in enumerate(audio_files):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 1] {af.name}")

            try:
                sm = sample_meta.get(af.name, {})
                prompt_variants = _prompt_variants_for_sample(sm, genre_ratio)
                expected_final_paths = [
                    _variant_final_path(out_path, af, variant)
                    for variant in prompt_variants
                ]
                if expected_final_paths and all(path.exists() for path in expected_final_paths):
                    logger.info("[Gary] Skipping (final variants exist): %s", af.name)
                    continue

                # 1. Load audio (stereo, 48 kHz)
                audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)
                audio = audio.unsqueeze(0).to(device=device, dtype=vae.dtype)

                # 2. VAE encode (tiled for long audio)
                with torch.no_grad():
                    target_latents = _tiled_vae_encode(vae, audio, dtype)

                # Free raw audio immediately -- no longer needed after VAE encode
                del audio

                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                # 3. Text encode
                caption = sm.get("caption", af.stem)
                lyrics = sm.get("lyrics", "[Instrumental]")

                with torch.no_grad():
                    lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, lyrics, device, dtype)

                for prompt_variant in prompt_variants:
                    use_genre = prompt_variant == "genre"
                    text_prompt = _build_simple_prompt(
                        sm, tag_position=tag_position, use_genre=use_genre
                    )

                    with torch.no_grad():
                        text_hs, text_mask = encode_text(
                            text_enc, tokenizer, text_prompt, device, dtype
                        )

                    # 4. Save intermediate
                    tmp_path = _variant_tmp_path(out_path, af, prompt_variant)
                    torch.save({
                        "target_latents": target_latents.squeeze(0).cpu(),
                        "attention_mask": attention_mask.squeeze(0).cpu(),
                        "text_hidden_states": text_hs.cpu(),
                        "text_attention_mask": text_mask.cpu(),
                        "lyric_hidden_states": lyric_hs.cpu(),
                        "lyric_attention_mask": lyric_mask.cpu(),
                        "silence_latent": silence_latent.cpu(),
                        "latent_length": latent_length,
                        "metadata": {
                            "audio_path": str(af),
                            "filename": af.name,
                            "caption": caption,
                            "lyrics": lyrics,
                            "duration": sm.get("duration", 0),
                            "bpm": sm.get("bpm"),
                            "keyscale": sm.get("keyscale", ""),
                            "timesignature": sm.get("timesignature", ""),
                            "genre": sm.get("genre", ""),
                            "is_instrumental": sm.get("is_instrumental", True),
                            "custom_tag": sm.get("custom_tag", ""),
                            "prompt_override": sm.get("prompt_override"),
                            "prompt_variant": prompt_variant,
                        },
                    }, tmp_path)
                    intermediates.append(tmp_path)
                    del text_hs, text_mask

                # Free GPU tensors from this iteration before the next one
                del target_latents, attention_mask
                del lyric_hs, lyric_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                logger.info(
                    "[Gary] Pass 1 OK: %s (%s)",
                    af.name,
                    ", ".join(prompt_variants),
                )

            except Exception as exc:
                failed += 1
                logger.error("[Gary] Pass 1 FAIL %s: %s", af.name, exc)

    finally:
        logger.info("[Side-Step] Unloading VAE + Text Encoder ...")
        unload_models(vae, text_enc, tokenizer, silence_latent)

    if progress_callback:
        progress_callback(total, total, "[Pass 1] Done")

    return intermediates, failed


# ---------------------------------------------------------------------------
# Pass 2 -- Heavy model (DIT encoder)
# ---------------------------------------------------------------------------

def _pass2_heavy(
    intermediates: List[Path],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[int, int]:
    """Run DIT encoder on intermediates and write final .pt files.

    Returns ``(processed_count, fail_count)``.
    """
    if not intermediates:
        return 0, 0

    from acestep.training_v2.model_loader import (
        load_decoder_for_training,
        unload_models,
        _resolve_dtype,
    )
    from acestep.training.dataset_builder_modules.preprocess_encoder import run_encoder
    from acestep.training.dataset_builder_modules.preprocess_context import build_context_latents

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 2/2: Loading DIT model (variant=%s) ...", variant)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)

    processed = 0
    failed = 0
    total = len(intermediates)

    try:
        for i, tmp_path in enumerate(intermediates):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            if progress_callback:
                progress_callback(i, total, f"[Pass 2] {tmp_path.stem}")

            try:
                data = torch.load(str(tmp_path), weights_only=False)

                # Move tensors directly to model device/dtype (single .to()
                # avoids creating throwaway intermediate GPU copies).
                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype

                text_hs = data["text_hidden_states"].to(model_device, dtype=model_dtype)
                text_mask = data["text_attention_mask"].to(model_device, dtype=model_dtype)
                lyric_hs = data["lyric_hidden_states"].to(model_device, dtype=model_dtype)
                lyric_mask = data["lyric_attention_mask"].to(model_device, dtype=model_dtype)
                silence_latent = data["silence_latent"].to(model_device, dtype=model_dtype)
                latent_length = data["latent_length"]

                # DIT encoder pass (adapter-agnostic: same tensors for
                # LoRA and LoKR -- only the adapter injection differs).
                encoder_hs, encoder_mask = run_encoder(
                    model,
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    device=str(model_device),
                    dtype=model_dtype,
                )

                # Free encoder inputs immediately after use
                del text_hs, text_mask, lyric_hs, lyric_mask

                # Build context latents (silence-based, standard text2music)
                if silence_latent.dim() == 2:
                    silence_latent = silence_latent.unsqueeze(0)

                context_latents = build_context_latents(
                    silence_latent, latent_length, str(model_device), model_dtype,
                )
                del silence_latent

                # Write final .pt  (strip ".tmp" from "song.tmp.pt" -> "song.pt")
                base_name = tmp_path.name.replace(".tmp.pt", ".pt")
                final_path = out_path / base_name
                meta = data["metadata"]
                torch.save({
                    "target_latents": data["target_latents"],
                    "attention_mask": data["attention_mask"],
                    "encoder_hidden_states": encoder_hs.squeeze(0).cpu(),
                    "encoder_attention_mask": encoder_mask.squeeze(0).cpu(),
                    "context_latents": context_latents.squeeze(0).cpu(),
                    "metadata": meta,
                }, final_path)

                # Free all GPU tensors and the loaded data dict before next iter
                del encoder_hs, encoder_mask, context_latents, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Remove intermediate
                tmp_path.unlink(missing_ok=True)

                processed += 1
                logger.info("[Gary] Pass 2 OK: %s", tmp_path.stem)

            except Exception as exc:
                failed += 1
                logger.error("[Gary] Pass 2 FAIL %s: %s", tmp_path.stem, exc)

    finally:
        logger.info("[Side-Step] Unloading DIT model ...")
        unload_models(model)

    if progress_callback:
        progress_callback(total, total, "[Pass 2] Done")

    return processed, failed


def _prompt_variants_for_sample(meta: Dict[str, Any], genre_ratio: int) -> List[str]:
    """Return prompt variants to encode for one sample.

    A positive genre ratio means Gary encodes the caption prompt and, when the
    sidecar has a distinct genre, a genre prompt for the same audio. The
    manifest then controls the effective caption/genre sampling ratio.
    """
    override = str(meta.get("prompt_override") or "").strip().lower()
    has_genre = _has_distinct_genre(meta)
    if override == "genre":
        return ["genre" if has_genre else "caption"]
    if override == "caption":
        return ["caption"]
    if genre_ratio >= 100 and has_genre:
        return ["genre"]
    variants = ["caption"]
    if genre_ratio > 0 and has_genre:
        variants.append("genre")
    return variants


def _has_distinct_genre(meta: Dict[str, Any]) -> bool:
    genre = str(meta.get("genre") or "").strip()
    if not genre:
        return False
    caption = str(meta.get("caption") or "").strip()
    return genre.casefold() != caption.casefold()


def _variant_tmp_path(out_path: Path, audio_path: Path, variant: str) -> Path:
    if variant == "caption":
        return out_path / f"{audio_path.stem}.tmp.pt"
    return out_path / f"{audio_path.stem}.{variant}.tmp.pt"


def _variant_final_path(out_path: Path, audio_path: Path, variant: str) -> Path:
    if variant == "caption":
        return out_path / f"{audio_path.stem}.pt"
    return out_path / f"{audio_path.stem}.{variant}.pt"


def _write_variant_manifest(
    *,
    out_path: Path,
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    ds_meta: Dict[str, Any],
) -> int:
    """Write one logical training row per audio track.

    Caption and genre tensors remain separate cache files.  The data loader
    chooses the genre path for a rotating subset of tracks each epoch, so a
    20% ratio does not multiply the number of training samples per epoch.
    """
    genre_ratio = int(ds_meta.get("genre_ratio", 0) or 0)
    manifest_samples: List[str] = []
    sample_groups: List[Dict[str, str]] = []
    tensor_files: List[str] = []
    genre_variant_tracks = 0

    for af in audio_files:
        variants = _prompt_variants_for_sample(sample_meta.get(af.name, {}), genre_ratio)
        caption_path = _variant_final_path(out_path, af, "caption")
        genre_path = _variant_final_path(out_path, af, "genre")
        has_caption = "caption" in variants and caption_path.exists()
        has_genre = "genre" in variants and genre_path.exists()

        if has_caption:
            group = {"path": caption_path.name}
            manifest_samples.append(caption_path.name)
            tensor_files.append(caption_path.name)
            if has_genre:
                group["genre_path"] = genre_path.name
                tensor_files.append(genre_path.name)
                genre_variant_tracks += 1
            sample_groups.append(group)
        elif has_genre:
            # Explicit genre overrides and 100% genre mode use the genre
            # tensor as the track's sole/default prompt.
            manifest_samples.append(genre_path.name)
            sample_groups.append({"path": genre_path.name})
            tensor_files.append(genre_path.name)

    metadata = dict(ds_meta)
    metadata.update(
        {
            "prompt_variant_strategy": "epoch_rotating_track_swap",
            "target_genre_ratio": genre_ratio,
            "genre_variant_tracks": genre_variant_tracks,
            "samples_per_epoch": len(sample_groups),
            "tensor_files": sorted(set(tensor_files)),
            "num_tensor_files": len(set(tensor_files)),
        }
    )
    manifest = {
        "metadata": metadata,
        "samples": manifest_samples,
        "sample_groups": sample_groups,
        "num_samples": len(manifest_samples),
    }
    manifest_path = out_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    logger.info(
        "[Gary] Wrote manifest: %d tracks per epoch from %d cached tensors "
        "(%d tracks can swap to genre; target %d%%)",
        len(manifest_samples),
        len(set(tensor_files)),
        genre_variant_tracks,
        genre_ratio,
    )
    return len(manifest_samples)
