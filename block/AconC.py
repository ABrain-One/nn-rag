# Auto-generated single-file for AconC
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- AconC (target) ----
class AconC(nn.Module):
    """
    ACON activation (activate or not) function.

    AconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x, beta is a learnable parameter
    See "Activate or Not: Learning Customized Activation" https://arxiv.org/pdf/2009.04759.pdf.
    """

    def __init__(self, c1):
        """Initializes AconC with learnable parameters p1, p2, and beta for channel-wise activation control."""
        super().__init__()
        self.p1 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, c1, 1, 1))
        self.beta = nn.Parameter(torch.ones(1, c1, 1, 1))

    def forward(self, x):
        """Applies AconC activation function with learnable parameters for channel-wise control on input tensor x."""
        dpx = (self.p1 - self.p2) * x
        return dpx * torch.sigmoid(self.beta * dpx) + self.p2 * x
