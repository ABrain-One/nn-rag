# Auto-generated single-file for BertOnlyNSPHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn

# ---- original imports from contributing modules ----
from torch import nn

# ---- BertOnlyNSPHead (target) ----
class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score
