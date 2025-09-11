# Auto-generated single-file for BertOutput
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
from torch import Tensor

# ---- mmdet.models.utils.vlfuse_helper.HFBertOutput ----
try:
    from transformers import BertConfig, BertPreTrainedModel
    from transformers.modeling_utils import apply_chunking_to_forward
    from transformers.models.bert.modeling_bert import \
        BertAttention as HFBertAttention
    from transformers.models.bert.modeling_bert import \
        BertIntermediate as HFBertIntermediate
    from transformers.models.bert.modeling_bert import \
        BertOutput as HFBertOutput
except ImportError:
    BertConfig = None
    BertPreTrainedModel = object
    apply_chunking_to_forward = None
    HFBertAttention = object
    HFBertIntermediate = object
    HFBertOutput = object

# ---- mmdet.models.utils.vlfuse_helper.MAX_CLAMP_VALUE ----
MAX_CLAMP_VALUE = 50000

# ---- mmdet.models.utils.vlfuse_helper.clamp_values ----
def clamp_values(vector: Tensor) -> Tensor:
    """Clamp the values of a vector to the range [-MAX_CLAMP_VALUE,
    MAX_CLAMP_VALUE].

    Args:
        vector (Tensor): Tensor of shape (N, C, H, W).

    Returns:
        Tensor: A Tensor of shape (N, C, H, W) with clamped values.
    """
    vector = torch.clamp(vector, min=-MAX_CLAMP_VALUE, max=MAX_CLAMP_VALUE)
    return vector

# ---- BertOutput (target) ----
class BertOutput(HFBertOutput):
    """Modified from transformers.models.bert.modeling_bert.BertOutput.

    Compared to the BertOutput of Huggingface, only add the clamp.
    """

    def forward(self, hidden_states: Tensor, input_tensor: Tensor) -> Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = clamp_values(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        hidden_states = clamp_values(hidden_states)
        return hidden_states
