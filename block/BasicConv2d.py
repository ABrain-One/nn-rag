# Auto-generated single-file for BasicConv2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Any

# ---- original imports from contributing modules ----

# ---- BasicConv2d (target) ----
class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)
