"""Padding helpers for handler batch preparation."""

import torch
from loguru import logger


class PaddingMixin:
    """Mixin containing repaint/lego padding helpers.

    Depends on host members:
    - Method: ``create_target_wavs`` (provided by ``TaskUtilsMixin`` in this decomposition).
    """

    def prepare_padding_info(
        self,
        actual_batch_size,
        processed_src_audio,
        audio_duration,
        repainting_start,
        repainting_end,
        is_repaint_task,
        is_lego_task,
        is_cover_task,
        can_use_repainting,
        is_complete_task=False,
    ):
        """Prepare padded target wavs and repaint coordinates for each batch item."""
        try:
            target_wavs_batch = []
            # Store padding info for each batch item to adjust repainting coordinates
            padding_info_batch = []
            for i in range(actual_batch_size):
                if processed_src_audio is not None:
                    if is_cover_task:
                        # Cover task: Use src_audio directly without padding
                        batch_target_wavs = processed_src_audio
                        padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
                    elif is_repaint_task or is_lego_task:
                        # Repaint/lego task: May need padding for outpainting
                        src_audio_duration = processed_src_audio.shape[-1] / 48000.0

                        # Determine actual end time
                        if repainting_end is None or repainting_end < 0:
                            actual_end = src_audio_duration
                        else:
                            actual_end = repainting_end

                        left_padding_duration = max(0, -repainting_start) if repainting_start is not None else 0
                        right_padding_duration = max(0, actual_end - src_audio_duration)

                        # Create padded audio
                        left_padding_frames = int(left_padding_duration * 48000)
                        right_padding_frames = int(right_padding_duration * 48000)

                        if left_padding_frames > 0 or right_padding_frames > 0:
                            # Pad the src audio
                            batch_target_wavs = torch.nn.functional.pad(
                                processed_src_audio, (left_padding_frames, right_padding_frames), "constant", 0
                            )
                        else:
                            batch_target_wavs = processed_src_audio

                        # Store padding info for coordinate adjustment
                        padding_info_batch.append(
                            {
                                "left_padding_duration": left_padding_duration,
                                "right_padding_duration": right_padding_duration,
                            }
                        )
                    elif is_complete_task:
                        # Complete task: pad source audio to the desired audio_duration if longer.
                        # The padded tensor (src + silence) gives the DiT harmonic/rhythmic context
                        # from the source while providing space to complete the arrangement.
                        src_audio_duration = processed_src_audio.shape[-1] / 48000.0
                        target_duration = (
                            float(audio_duration)
                            if audio_duration is not None and float(audio_duration) > src_audio_duration
                            else src_audio_duration
                        )
                        right_padding_frames = int(max(0, target_duration - src_audio_duration) * 48000)
                        if right_padding_frames > 0:
                            batch_target_wavs = torch.nn.functional.pad(
                                processed_src_audio, (0, right_padding_frames), "constant", 0
                            )
                        else:
                            batch_target_wavs = processed_src_audio
                        padding_info_batch.append(
                            {"left_padding_duration": 0.0, "right_padding_duration": target_duration - src_audio_duration}
                        )
                    else:
                        # Other tasks: Use src_audio directly without padding
                        batch_target_wavs = processed_src_audio
                        padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
                else:
                    padding_info_batch.append({"left_padding_duration": 0.0, "right_padding_duration": 0.0})
                    if audio_duration is not None and float(audio_duration) > 0:
                        batch_target_wavs = self.create_target_wavs(float(audio_duration))
                    else:
                        import random

                        random_duration = random.uniform(10.0, 120.0)
                        batch_target_wavs = self.create_target_wavs(random_duration)
                target_wavs_batch.append(batch_target_wavs)

            # Stack target_wavs into batch tensor
            # Ensure all tensors have the same shape by padding to max length
            max_frames = max(wav.shape[-1] for wav in target_wavs_batch)
            padded_target_wavs = []
            for wav in target_wavs_batch:
                if wav.shape[-1] < max_frames:
                    pad_frames = max_frames - wav.shape[-1]
                    padded_wav = torch.nn.functional.pad(wav, (0, pad_frames), "constant", 0)
                    padded_target_wavs.append(padded_wav)
                else:
                    padded_target_wavs.append(wav)

            target_wavs_tensor = torch.stack(padded_target_wavs, dim=0)  # [batch_size, 2, frames]

            if can_use_repainting:
                if is_complete_task and processed_src_audio is not None:
                    # Complete task: outpainting — preserve the source audio as-is and
                    # generate from the end of the source to the desired duration.
                    # repainting_start = src_audio_duration → source is locked in as context
                    # repainting_end   = target_duration    → generate until end of desired output
                    # conditioning_masks will zero out src_latent[src_end:target_end] and set
                    # the chunk_mask True only for that region, so the DiT generates only the
                    # continuation while the original source frames remain untouched.
                    src_audio_duration = processed_src_audio.shape[-1] / 48000.0
                    target_duration = (
                        float(audio_duration)
                        if audio_duration is not None and float(audio_duration) > src_audio_duration
                        else src_audio_duration
                    )
                    repainting_start_batch = [src_audio_duration] * actual_batch_size
                    repainting_end_batch = [target_duration] * actual_batch_size
                else:
                    # Repaint / lego task: Set repainting parameters
                    if repainting_start is None:
                        repainting_start_batch = None
                    elif isinstance(repainting_start, (int, float)):
                        if processed_src_audio is not None:
                            adjusted_start = repainting_start + padding_info_batch[0]["left_padding_duration"]
                            repainting_start_batch = [adjusted_start] * actual_batch_size
                        else:
                            repainting_start_batch = [repainting_start] * actual_batch_size
                    else:
                        # List input - adjust each item
                        repainting_start_batch = []
                        for i in range(actual_batch_size):
                            if processed_src_audio is not None:
                                adjusted_start = repainting_start[i] + padding_info_batch[i]["left_padding_duration"]
                                repainting_start_batch.append(adjusted_start)
                            else:
                                repainting_start_batch.append(repainting_start[i])

                    # Handle repainting_end - use src audio duration if not specified or negative
                    if processed_src_audio is not None:
                        # If src audio is provided, use its duration as default end
                        src_audio_duration = processed_src_audio.shape[-1] / 48000.0
                        if repainting_end is None or repainting_end < 0:
                            if is_lego_task:
                                # For lego with full-audio mask, leave repainting_end as None so
                                # conditioning_masks takes the full-mask branch, which preserves
                                # src_latents as the full source audio context. Converting -1
                                # to the audio duration routes through the repainting branch
                                # which silences the entire src_latents tensor, giving the DiT
                                # zero harmonic/rhythmic context from the source.
                                repainting_end_batch = None
                            else:
                                # Use src audio duration (before padding), then adjust for padding
                                adjusted_end = src_audio_duration + padding_info_batch[0]["left_padding_duration"]
                                repainting_end_batch = [adjusted_end] * actual_batch_size
                        else:
                            # Adjust repainting_end to be relative to padded audio
                            adjusted_end = repainting_end + padding_info_batch[0]["left_padding_duration"]
                            repainting_end_batch = [adjusted_end] * actual_batch_size
                    else:
                        # No src audio - repainting doesn't make sense without it
                        if repainting_end is None or repainting_end < 0:
                            repainting_end_batch = None
                        elif isinstance(repainting_end, (int, float)):
                            repainting_end_batch = [repainting_end] * actual_batch_size
                        else:
                            # List input - adjust each item
                            repainting_end_batch = []
                            for i in range(actual_batch_size):
                                repainting_end_batch.append(repainting_end[i])
            else:
                # All other tasks (cover, text2music, extract): No repainting
                repainting_start_batch = None
                repainting_end_batch = None

            return repainting_start_batch, repainting_end_batch, target_wavs_tensor
        except (TypeError, ValueError, RuntimeError, IndexError):
            logger.exception("[prepare_padding_info] Error preparing padding information")
            fallback = torch.stack([self.create_target_wavs(30.0) for _ in range(actual_batch_size)], dim=0)
            return None, None, fallback
