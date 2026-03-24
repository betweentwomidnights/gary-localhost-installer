# pyre-strict

import typing as tp

import torch


# Based on code from https://github.com/pix2pixzero/pix2pix-zero
@torch.enable_grad()
def noise_regularization(
    e_t: torch.Tensor,
    noise_pred_optimal: torch.Tensor,
    lambda_kl: float,
    lambda_ac: float,
    num_reg_steps: int,
    num_ac_rolls: int,
    generator: tp.Optional[torch._C.Generator] = None,
) -> torch.Tensor:
    should_move_back_to_cpu = e_t.device.type == "mps"
    # print(should_move_back_to_cpu)
    if should_move_back_to_cpu:
        e_t = e_t.to("cpu")
        noise_pred_optimal = noise_pred_optimal.to("cpu")
    for _outer in range(num_reg_steps):
        if lambda_kl > 0:
            _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
            l_kld = patchify_latents_kl_divergence(_var, noise_pred_optimal)
            l_kld.backward()
            _grad = _var.grad.detach()  # pyre-ignore
            _grad = torch.clip(_grad, -100, 100)
            e_t = e_t - lambda_kl * _grad
        if lambda_ac > 0:
            for _inner in range(num_ac_rolls):
                _var = torch.autograd.Variable(e_t.detach().clone(), requires_grad=True)
                l_ac = auto_corr_loss(_var.unsqueeze(1), generator=generator)
                l_ac.backward()  # pyre-ignore
                _grad = _var.grad.detach() / num_ac_rolls
                e_t = e_t - lambda_ac * _grad
        e_t = e_t.detach()

    return e_t if not should_move_back_to_cpu else e_t.to("mps")


# Based on code from https://github.com/pix2pixzero/pix2pix-zero
def auto_corr_loss(
    x: torch.Tensor,
    random_shift: bool = True,
    generator: tp.Optional[torch._C.Generator] = None,
) -> tp.Union[float, torch.Tensor]:
    B, C, H, W = x.shape
    assert B == 1
    x = x.squeeze(0)
    # x must be shape [C,H,W] now
    reg_loss = 0.0
    for ch_idx in range(x.shape[0]):
        noise = x[ch_idx][None, None, :, :]
        while True:
            if random_shift:
                roll_amount = torch.randint(
                    0, noise.shape[2] // 2, (1,), generator=generator
                ).item()
            else:
                roll_amount = 1
            reg_loss += torch.pow(
                (noise * torch.roll(noise, shifts=roll_amount, dims=2)).mean(), 2  # pyre-ignore
            )
            reg_loss += torch.pow(
                (noise * torch.roll(noise, shifts=roll_amount, dims=3)).mean(), 2  # pyre-ignore
            )
            if noise.shape[2] <= 8:
                break
            noise = torch.nn.functional.avg_pool2d(noise, kernel_size=2)
    return reg_loss


def patchify_latents_kl_divergence(
    x0: torch.Tensor, x1: torch.Tensor, patch_size: int = 4, num_channels: int = 4
) -> torch.Tensor:

    def patchify_tensor(input_tensor: torch.Tensor) -> torch.Tensor:
        patches = (
            input_tensor.unfold(1, patch_size, patch_size)
            .unfold(2, patch_size, patch_size)
            .unfold(3, patch_size, patch_size)
        )
        patches = patches.contiguous().view(-1, num_channels, patch_size, patch_size)
        return patches

    x0 = patchify_tensor(x0)
    x1 = patchify_tensor(x1)

    kl = latents_kl_divergence(x0, x1).sum()
    return kl


def latents_kl_divergence(x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
    EPSILON = 1e-6
    x0 = x0.view(x0.shape[0], x0.shape[1], -1)
    x1 = x1.view(x1.shape[0], x1.shape[1], -1)
    mu0 = x0.mean(dim=-1)
    mu1 = x1.mean(dim=-1)
    var0 = x0.var(dim=-1)
    var1 = x1.var(dim=-1)
    kl = (
        torch.log((var1 + EPSILON) / (var0 + EPSILON))
        + (var0 + torch.pow((mu0 - mu1), 2)) / (var1 + EPSILON)
        - 1
    )
    kl = torch.abs(kl).sum(dim=-1)
    return kl
