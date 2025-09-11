# Auto-generated single-file for BidirectionalLSTM
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn

# ---- BidirectionalLSTM (target) ----
class BidirectionalLSTM(nn.Module):

    def __init__(self, nIn, nHidden, nOut):
        super().__init__()

        self.rnn = nn.LSTM(nIn, nHidden, bidirectional=True)
        self.embedding = nn.Linear(nHidden * 2, nOut)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, nOut]
        output = output.view(T, b, -1)

        return output
