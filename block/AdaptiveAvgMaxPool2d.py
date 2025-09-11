# Auto-generated single-file for AdaptiveAvgMaxPool2d
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Union

# ---- timm.layers.adaptive_avgmax_pool._int_tuple_2_t ----
_int_tuple_2_t = Union[int, Tuple[int, int]]

# ---- timm.layers.adaptive_avgmax_pool.adaptive_avgmax_pool2d ----
def adaptive_avgmax_pool2d(x, output_size: _int_tuple_2_t = 1):
    x_avg = F.adaptive_avg_pool2d(x, output_size)
    x_max = F.adaptive_max_pool2d(x, output_size)
    return 0.5 * (x_avg + x_max)

# ---- AdaptiveAvgMaxPool2d (target) ----
class AdaptiveAvgMaxPool2d(nn.Module):
    def __init__(self, output_size: _int_tuple_2_t = 1):
        super(AdaptiveAvgMaxPool2d, self).__init__()
        self.output_size = output_size

    def forward(self, x):
        return adaptive_avgmax_pool2d(x, self.output_size)
