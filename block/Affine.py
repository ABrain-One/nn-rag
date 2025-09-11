# Auto-generated single-file for Affine
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn

# ---- original imports from contributing modules ----

# ---- Affine (target) ----
class Affine(nn.Module):
    """Affine transformation layer."""

    def __init__(self, dim: int) -> None:
        """Initialize Affine layer.

        Args:
            dim: Dimension of features.
        """
        super().__init__()
        self.alpha = nn.Parameter(torch.ones((1, 1, dim)))
        self.beta = nn.Parameter(torch.zeros((1, 1, dim)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply affine transformation."""
        return torch.addcmul(self.beta, self.alpha, x)
