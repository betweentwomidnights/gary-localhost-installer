"""Helpers for crossing the sampler-to-autoencoder dtype/device boundary."""


def align_latents_for_decode(latents, pretransform, on_cast=None):
    """Move sampler output to the dtype and device expected by the decoder."""
    decoder_param = next(pretransform.parameters(), None)
    if decoder_param is None:
        return latents

    if latents.dtype != decoder_param.dtype or latents.device != decoder_param.device:
        if on_cast is not None:
            on_cast(
                latents.device,
                latents.dtype,
                decoder_param.device,
                decoder_param.dtype,
            )
        latents = latents.to(
            device=decoder_param.device,
            dtype=decoder_param.dtype,
        )
    return latents
