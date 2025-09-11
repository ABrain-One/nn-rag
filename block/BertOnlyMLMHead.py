# Auto-generated single-file for BertOnlyMLMHead
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
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

# ---- mmpretrain.models.multimodal.blip.language_model.BertPredictionHeadTransform ----
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

# ---- mmpretrain.models.multimodal.blip.language_model.BertLMPredictionHead ----
class BertLMPredictionHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False)

        self.bias = nn.Parameter(torch.zeros(config.vocab_size))

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states

# ---- BertOnlyMLMHead (target) ----
class BertOnlyMLMHead(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.predictions = BertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores
