# Auto-generated single-file for BatchNorm1dEdge
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch

# ---- original imports from contributing modules ----

# ---- BatchNorm1dEdge (target) ----
class BatchNorm1dEdge(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.bn = torch.nn.BatchNorm1d(channels, eps=1e-5, momentum=0.1)

    def forward(self, batch):
        batch.edge_attr = self.bn(batch.edge_attr)
        return batch
