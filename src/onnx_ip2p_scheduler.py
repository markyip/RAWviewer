"""Euler-ancestral scheduler for the local InstructPix2Pix provider, in numpy.

The reference implementation lives in ``diffusers``, which is a torch
dependency this app does not have and does not want -- the whole ML stack is
ONNX Runtime plus numpy. The scheduler is pure arithmetic on the noise
schedule, so it is reimplemented here rather than dragging torch in for it.

Matches ``EulerAncestralDiscreteScheduler`` with the ``scaled_linear`` beta
schedule and epsilon prediction, which is what InstructPix2Pix was trained
and published with.
"""

from __future__ import annotations

import json
import os
from typing import Optional

import numpy as np


class EulerAncestralScheduler:
    """Epsilon-prediction Euler-ancestral sampling over a sigma schedule."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.00085,
        beta_end: float = 0.012,
        beta_schedule: str = "scaled_linear",
        steps_offset: int = 1,
    ):
        self.num_train_timesteps = int(num_train_timesteps)
        self.steps_offset = int(steps_offset)

        if beta_schedule == "linear":
            betas = np.linspace(beta_start, beta_end, self.num_train_timesteps)
        else:
            # scaled_linear: linear in sqrt(beta), the schedule Stable
            # Diffusion and everything derived from it uses.
            betas = (
                np.linspace(
                    beta_start**0.5, beta_end**0.5, self.num_train_timesteps
                )
                ** 2
            )
        betas = betas.astype(np.float64)
        alphas = 1.0 - betas
        self.alphas_cumprod = np.cumprod(alphas, axis=0)

        sigmas = np.sqrt((1.0 - self.alphas_cumprod) / self.alphas_cumprod)
        self._train_sigmas = sigmas

        self.timesteps = np.array([], dtype=np.float32)
        self.sigmas = np.array([], dtype=np.float32)

    @classmethod
    def from_config_file(cls, path: str) -> "EulerAncestralScheduler":
        """Build from the scheduler_config.json shipped beside the graphs.

        Reading it keeps the schedule tied to the downloaded weights instead
        of assuming the values the reference model happened to use.
        """
        cfg = {}
        try:
            if path and os.path.isfile(path):
                with open(path, "r", encoding="utf-8") as fh:
                    cfg = json.load(fh)
        except Exception:
            cfg = {}
        return cls(
            num_train_timesteps=int(cfg.get("num_train_timesteps", 1000)),
            beta_start=float(cfg.get("beta_start", 0.00085)),
            beta_end=float(cfg.get("beta_end", 0.012)),
            beta_schedule=str(cfg.get("beta_schedule", "scaled_linear")),
            steps_offset=int(cfg.get("steps_offset", 1)),
        )

    # -- schedule -----------------------------------------------------

    def set_timesteps(self, num_inference_steps: int) -> None:
        n = max(1, int(num_inference_steps))
        timesteps = np.linspace(
            0, self.num_train_timesteps - 1, n, dtype=np.float32
        )[::-1].copy()

        sigmas = np.interp(
            timesteps,
            np.arange(self.num_train_timesteps, dtype=np.float32),
            self._train_sigmas.astype(np.float32),
        )
        # Trailing zero: the final step lands on a clean latent.
        self.sigmas = np.concatenate([sigmas, [0.0]]).astype(np.float32)
        self.timesteps = timesteps

    @property
    def init_noise_sigma(self) -> float:
        return float(self.sigmas[0]) if len(self.sigmas) else 1.0

    def scale_model_input(self, sample: np.ndarray, index: int) -> np.ndarray:
        """Normalise the latent so the UNet sees unit-variance input."""
        sigma = float(self.sigmas[index])
        return (sample / ((sigma**2 + 1) ** 0.5)).astype(np.float32)

    def step(
        self,
        model_output: np.ndarray,
        index: int,
        sample: np.ndarray,
        generator: Optional[np.random.Generator] = None,
    ) -> np.ndarray:
        """One ancestral Euler step: denoise, then re-inject a little noise."""
        sigma = float(self.sigmas[index])
        sigma_to = float(self.sigmas[index + 1])

        # Epsilon prediction -> the model's estimate of the clean latent.
        pred_original = sample - sigma * model_output

        sigma_up_sq = (
            (sigma_to**2) * (sigma**2 - sigma_to**2) / (sigma**2)
            if sigma > 0
            else 0.0
        )
        sigma_up = float(np.sqrt(max(sigma_up_sq, 0.0)))
        sigma_down = float(np.sqrt(max(sigma_to**2 - sigma_up**2, 0.0)))

        derivative = (sample - pred_original) / sigma if sigma > 0 else np.zeros_like(sample)
        prev = sample + derivative * (sigma_down - sigma)

        if sigma_up > 0:
            rng = generator if generator is not None else np.random.default_rng()
            noise = rng.standard_normal(size=sample.shape).astype(np.float32)
            prev = prev + noise * sigma_up
        return prev.astype(np.float32)
