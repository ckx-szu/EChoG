import torch
import torch.nn as nn

class AxialLatentBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, dropout: float) -> None:
        super().__init__()
        self.spatial = nn.Sequential(
            nn.Conv3d(
                hidden, hidden, (3, 3, 3), padding=1, groups=hidden
            ),
            nn.GroupNorm(min(16, hidden), hidden),
            nn.SiLU(),
            nn.Conv3d(hidden, hidden, 1),
        )
        self.temporal_norm = nn.LayerNorm(hidden)
        self.temporal_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.output_norm = nn.GroupNorm(min(16, hidden), hidden)
        self.output = nn.Sequential(
            nn.Conv3d(hidden, hidden * 2, 1),
            nn.SiLU(),
            nn.Conv3d(hidden * 2, hidden, 1),
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.spatial(value)
        batch, channels, frames, height, width = value.shape
        timeline = value.permute(0, 3, 4, 2, 1).reshape(
            batch * height * width, frames, channels
        )
        normalized = self.temporal_norm(timeline)
        update, _ = self.temporal_attention(
            normalized, normalized, normalized, need_weights=False
        )
        value = (timeline + update).reshape(
            batch, height, width, frames, channels
        ).permute(0, 4, 3, 1, 2)
        return value + self.output(self.output_norm(value))