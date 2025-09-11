# Auto-generated single-file for BertPredictionHeadTransform
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch.nn as nn

# ---- mmpretrain.models.multimodal.blip.language_model.ACT2FN ----
try:
    from transformers.activations import ACT2FN
    from transformers.modeling_outputs import (
        BaseModelOutputWithPastAndCrossAttentions,
        BaseModelOutputWithPoolingAndCrossAttentions,
        CausalLMOutputWithCrossAttentions)
    from transformers.modeling_utils import (PreTrainedModel,
                                             apply_chunking_to_forward,
                                             find_pruneable_heads_and_indices,
                                             prune_linear_layer)
    from transformers.models.bert.configuration_bert import BertConfig
except:
    ACT2FN = None
    BaseModelOutputWithPastAndCrossAttentions = None
    BaseModelOutputWithPoolingAndCrossAttentions = None
    CausalLMOutputWithCrossAttentions = None
    PreTrainedModel = None
    apply_chunking_to_forward = None
    find_pruneable_heads_and_indices = None
    prune_linear_layer = None
    BertConfig = None

# ---- BertPredictionHeadTransform (target) ----
class BertPredictionHeadTransform(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states
