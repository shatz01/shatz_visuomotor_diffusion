"""Build and visualize the cosine diffusion noise schedule."""

from pathlib import Path

import matplotlib.pyplot as plt
import torch

"""
βt       new noise added now
αt       previous state retained now
ᾱt       original trajectory still remaining
√ᾱt      clean-trajectory mixing coefficient
√(1-ᾱt) Gaussian-noise mixing coefficient
log-SNR  whether signal or noise currently dominates
"""

def _get_alpha_bar(
    num_steps: int,
    offset: float = 0.008,
) -> torch.Tensor:
    if num_steps < 1:
        raise ValueError("num_steps must be positive")

    # With 100 diffusion steps, these are the 101 boundaries from 0 to 100.
    steps = torch.arange(num_steps + 1, dtype=torch.float64)  # shape [101]: [0., 1., 2., ..., 99., 100.]
    progress = steps / num_steps  # shape [101]: [0.00, 0.01, 0.02, ..., 0.99, 1.00]
    alpha_bar = torch.cos(
        (progress + offset) / (1 + offset) * torch.pi / 2
    ).square()  # [0.99984, 0.99921, 0.99810, ..., 0.0002428, 3.7494e-33]
    # Normalization makes the clean boundary contain exactly 100% signal power.
    alpha_bar = alpha_bar / alpha_bar[0]  # [1.00000, 0.99937, 0.99825, ..., 0.0002429, 3.7500e-33]
    return alpha_bar


def cosine_beta_schedule(
    num_steps: int,
    offset: float = 0.008,
) -> torch.Tensor:
    alpha_bar = _get_alpha_bar(num_steps, offset)  # shape [101]
    # Each beta is the extra noise needed to move between adjacent boundaries.
    betas = 1 - alpha_bar[1:] / alpha_bar[:-1]  # shape [100]: [0.0006313, 0.0011169, ..., 0.74994, 1.00000]
    return betas.clamp(max=0.999).float()


def plot_schedule(
    num_steps: int,
    output_path: Path = Path("outputs/cosine_beta_schedule.png"),
    offset: float = 0.008,
) -> None:
    """Plot the quantities used to construct and apply the noise schedule."""
    ideal_alpha_bar = _get_alpha_bar(num_steps, offset)[1:].float()
    betas = cosine_beta_schedule(num_steps, offset)
    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    noise_power = 1 - alpha_bars
    signal_coefficient = torch.sqrt(alpha_bars)
    noise_coefficient = torch.sqrt(noise_power)
    log_snr = torch.log(alpha_bars / noise_power)
    timesteps = torch.arange(num_steps)

    figure, axes = plt.subplots(2, 2, figsize=(11, 8))

    axes[0, 0].plot(timesteps, alpha_bars, label="signal power $\\bar{\\alpha}_t$")
    axes[0, 0].plot(timesteps, noise_power, label="noise power $1-\\bar{\\alpha}_t$")
    axes[0, 0].plot(
        timesteps,
        ideal_alpha_bar,
        linestyle="--",
        alpha=0.7,
        label="ideal cosine boundary",
    )
    axes[0, 0].set_title("Cumulative signal and noise power")
    axes[0, 0].legend()

    axes[0, 1].plot(timesteps, betas, label="$\\beta_t$ (noise added)")
    axes[0, 1].plot(timesteps, alphas, label="$\\alpha_t=1-\\beta_t$")
    axes[0, 1].set_title("Per-step schedule")
    axes[0, 1].legend()

    axes[1, 0].plot(
        timesteps,
        signal_coefficient,
        label="$\\sqrt{\\bar{\\alpha}_t}$ (clean trajectory)",
    )
    axes[1, 0].plot(
        timesteps,
        noise_coefficient,
        label="$\\sqrt{1-\\bar{\\alpha}_t}$ (Gaussian noise)",
    )
    axes[1, 0].set_title("Coefficients in the forward-noising equation")
    axes[1, 0].legend()

    axes[1, 1].plot(timesteps, log_snr)
    axes[1, 1].axhline(0, color="black", linewidth=1, linestyle="--")
    axes[1, 1].set_title("Log signal-to-noise ratio")

    for axis in axes.flat:
        axis.set_xlabel("Diffusion timestep")
        axis.grid(alpha=0.25)

    figure.suptitle(f"Cosine diffusion schedule ({num_steps} steps)")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    print(f"Schedule plot: {output_path.resolve()}")


if __name__ == "__main__":
    plot_schedule(num_steps=100)
