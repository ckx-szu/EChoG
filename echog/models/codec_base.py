from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MotionCodecV2Config:
    base_channels: int = 32
    latent_dim: int = 48
    temporal_stride: int = 2
    latent_mode: str = "continuous"
    fsq_levels: int = 7
    max_displacement: float = 0.30
    residual_scale: float = 0.12
    kl_weight: float = 1e-4

    def to_dict(self) -> dict:
        return asdict(self)


class ConvBlock2d(nn.Module):
    def __init__(self, inputs: int, outputs: int, stride: int = 1) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, stride=stride, padding=1),
            nn.GroupNorm(min(8, outputs), outputs),
            nn.SiLU(),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.GroupNorm(min(8, outputs), outputs),
            nn.SiLU(),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


def warp_first(first: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    batch, frames = flow.shape[:2]
    height, width = first.shape[-2:]
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, height, device=first.device, dtype=first.dtype),
        torch.linspace(-1, 1, width, device=first.device, dtype=first.dtype),
        indexing="ij",
    )
    base = torch.stack((x, y), dim=-1)[None, None]
    grid = base + flow.permute(0, 1, 3, 4, 2)
    source = first[:, None].expand(-1, frames, -1, -1, -1)
    return F.grid_sample(
        source.reshape(batch * frames, 1, height, width),
        grid.reshape(batch * frames, height, width, 2),
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    ).reshape(batch, frames, 1, height, width)


class CausalTemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int) -> None:
        super().__init__()
        self.padding = 2 * dilation
        self.depthwise = nn.Conv3d(
            channels,
            channels,
            (3, 1, 1),
            dilation=(dilation, 1, 1),
            groups=channels,
        )
        self.pointwise = nn.Conv3d(channels, channels, 1)
        self.norm = nn.GroupNorm(min(8, channels), channels)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        padded = F.pad(values, (0, 0, 0, 0, self.padding, 0))
        update = self.pointwise(F.silu(self.norm(self.depthwise(padded))))
        return values + update


def causal_average(values: torch.Tensor, stride: int) -> torch.Tensor:
    if stride <= 1:
        return values
    padded = F.pad(values, (0, 0, 0, 0, stride - 1, 0))
    return F.avg_pool3d(padded, (stride, 1, 1), (stride, 1, 1))


class ScalarFSQ(nn.Module):
    def __init__(self, levels: int) -> None:
        super().__init__()
        self.levels = int(levels)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        bounded = torch.tanh(values)
        scale = (self.levels - 1) / 2
        ids = torch.round((bounded + 1) * scale).clamp(0, self.levels - 1)
        quantized = ids / scale - 1
        return bounded + (quantized - bounded).detach(), ids.long()


class GenerationFriendlyMotionCodec(nn.Module):
    """R8 bottom codec weights, adapted to unknown reference position. No forced frame-zero identity."""

    def __init__(self, config: MotionCodecV2Config) -> None:
        super().__init__()
        if config.latent_mode not in {"continuous", "fsq"}:
            raise ValueError(f"unknown latent mode: {config.latent_mode}")
        self.config = config
        base, latent = config.base_channels, config.latent_dim
        self.frame_encoder = nn.Sequential(
            ConvBlock2d(2, base, 2),
            ConvBlock2d(base, base * 2, 2),
            ConvBlock2d(base * 2, latent, 2),
        )
        self.temporal = nn.Sequential(
            CausalTemporalBlock(latent, 1),
            CausalTemporalBlock(latent, 2),
            CausalTemporalBlock(latent, 4),
        )
        output_channels = (
            latent * 2 if config.latent_mode == "continuous" else latent
        )
        self.posterior = nn.Conv3d(latent, output_channels, 1)
        self.fsq = ScalarFSQ(config.fsq_levels)
        self.first_encoder = nn.Sequential(
            ConvBlock2d(1, base, 2),
            ConvBlock2d(base, base * 2, 2),
            ConvBlock2d(base * 2, latent, 2),
        )
        self.fusion = nn.Sequential(
            nn.Conv3d(latent * 2, latent, 3, padding=1),
            nn.GroupNorm(min(8, latent), latent),
            nn.SiLU(),
            CausalTemporalBlock(latent, 1),
            CausalTemporalBlock(latent, 2),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, base * 2, 4, 2, 1),
            nn.GroupNorm(min(8, base * 2), base * 2),
            nn.SiLU(),
            nn.ConvTranspose2d(base * 2, base, 4, 2, 1),
            nn.GroupNorm(min(8, base), base),
            nn.SiLU(),
            nn.ConvTranspose2d(base, 4, 4, 2, 1),
        )

    def encode(
        self, video: torch.Tensor, first_frame: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        batch, _, frames, height, width = video.shape
        repeated = first_frame[:, :, None].expand(-1, -1, frames, -1, -1)
        source = torch.cat((video, video - repeated), dim=1)
        encoded = self.frame_encoder(
            source.permute(0, 2, 1, 3, 4).reshape(
                batch * frames, 2, height, width
            )
        )
        encoded = encoded.reshape(
            batch,
            frames,
            self.config.latent_dim,
            encoded.shape[-2],
            encoded.shape[-1],
        ).permute(0, 2, 1, 3, 4)
        encoded = causal_average(
            self.temporal(encoded), self.config.temporal_stride
        )
        posterior = self.posterior(encoded)
        if self.config.latent_mode == "continuous":
            mean, logvar = posterior.chunk(2, dim=1)
            logvar = logvar.clamp(-8, 4)
            latent = mean
            regularization = -0.5 * (
                1 + logvar - mean.square() - logvar.exp()
            ).mean()
            ids = torch.empty(0, dtype=torch.long, device=video.device)
        else:
            mean, logvar = posterior, torch.zeros_like(posterior)
            latent, ids = self.fsq(posterior)
            regularization = (posterior - latent.detach()).square().mean()
        
        return {
            "latent": latent,
            "mean": mean,
            "logvar": logvar,
            "ids": ids,
            "regularization": regularization,
        }

    def decode(
        self, first_frame: torch.Tensor, latent: torch.Tensor, frames: int
    ) -> dict[str, torch.Tensor]:
        batch, _, height, width = first_frame.shape
        appearance = self.first_encoder(first_frame)
        motion = F.interpolate(
            latent,
            size=(frames, appearance.shape[-2], appearance.shape[-1]),
            mode="trilinear",
            align_corners=False,
        )
        appearance = appearance[:, :, None].expand(-1, -1, frames, -1, -1)
        fused = self.fusion(torch.cat((motion, appearance), dim=1))
        decoded = self.decoder(
            fused.permute(0, 2, 1, 3, 4).reshape(
                batch * frames,
                self.config.latent_dim,
                fused.shape[-2],
                fused.shape[-1],
            )
        ).reshape(batch, frames, 4, height, width)
        flow = torch.tanh(decoded[:, :, :2]) * self.config.max_displacement
        residual = (
            torch.tanh(decoded[:, :, 2:3]) * self.config.residual_scale
        )
        gate = torch.sigmoid(decoded[:, :, 3:4])
        warped = warp_first(first_frame, flow)
        reconstruction = (warped + gate * residual).clamp(0, 1)
        return {
            "video": reconstruction.permute(0, 2, 1, 3, 4),
            "warped": warped.permute(0, 2, 1, 3, 4),
            "flow": flow,
            "residual": residual,
        }

