# Auto-generated single-file for BertLayer
# Dependencies are emitted in topological order (utilities first).
# Standard library and external imports
import torch
import torch.nn as nn
import math

# ---- mmpretrain.models.multimodal.blip.language_model.BertSelfAttention ----
class BertSelfAttention(nn.Module):

    def __init__(self, config, is_cross_attention):
        super().__init__()
        self.config = config
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(
                config, 'embedding_size'):
            raise ValueError(
                'The hidden size (%d) is not a multiple of the number of attention '
                'heads (%d)' %
                (config.hidden_size, config.num_attention_heads))

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size /
                                       config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        if is_cross_attention:
            self.key = nn.Linear(config.encoder_width, self.all_head_size)
            self.value = nn.Linear(config.encoder_width, self.all_head_size)
        else:
            self.key = nn.Linear(config.hidden_size, self.all_head_size)
            self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config,
                                               'position_embedding_type',
                                               'absolute')
        if (self.position_embedding_type == 'relative_key'
                or self.position_embedding_type == 'relative_key_query'):
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(
                2 * config.max_position_embeddings - 1,
                self.attention_head_size)
        self.save_attention = False

    def save_attn_gradients(self, attn_gradients):
        self.attn_gradients = attn_gradients

    def get_attn_gradients(self):
        return self.attn_gradients

    def save_attention_map(self, attention_map):
        self.attention_map = attention_map

    def get_attention_map(self):
        return self.attention_map

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention:
            key_layer = self.transpose_for_scores(
                self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(
                self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer,
                                        key_layer.transpose(-1, -2))

        if (self.position_embedding_type == 'relative_key'
                or self.position_embedding_type == 'relative_key_query'):
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(
                seq_length, dtype=torch.long,
                device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(
                distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(
                dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == 'relative_key':
                relative_position_scores = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == 'relative_key_query':
                relative_position_scores_query = torch.einsum(
                    'bhld,lrd->bhlr', query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum(
                    'bhrd,lrd->bhlr', key_layer, positional_embedding)
                attention_scores = (
                    attention_scores + relative_position_scores_query +
                    relative_position_scores_key)

        attention_scores = attention_scores / math.sqrt(
            self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        if is_cross_attention and self.save_attention:
            self.save_attention_map(attention_probs)
            attention_probs.register_hook(self.save_attn_gradients)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs_dropped = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs_dropped = attention_probs_dropped * head_mask

        context_layer = torch.matmul(attention_probs_dropped, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (
            self.all_head_size, )
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = ((context_layer, attention_probs) if output_attentions else
                   (context_layer, ))

        outputs = outputs + (past_key_value, )
        return outputs

# ---- mmpretrain.models.multimodal.blip.language_model.BertOutput ----
class BertOutput(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

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

# ---- mmpretrain.models.multimodal.blip.language_model.BertSelfOutput ----
class BertSelfOutput(nn.Module):

    def __init__(self, config, twin=False, merge=False):
        super().__init__()
        self.LayerNorm = nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        if twin:
            self.dense0 = nn.Linear(config.hidden_size, config.hidden_size)
            self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        else:
            self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if merge:
            self.act = ACT2FN[config.hidden_act]
            self.merge_layer = nn.Linear(config.hidden_size * 2,
                                         config.hidden_size)
            self.merge = True
        else:
            self.merge = False

    def forward(self, hidden_states, input_tensor):
        if type(hidden_states) == list:
            hidden_states0 = self.dense0(hidden_states[0])
            hidden_states1 = self.dense1(hidden_states[1])
            if self.merge:
                hidden_states = self.merge_layer(
                    torch.cat([hidden_states0, hidden_states1], dim=-1))
            else:
                hidden_states = (hidden_states0 + hidden_states1) / 2
        else:
            hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# ---- mmpretrain.models.multimodal.blip.language_model.BertIntermediate ----
class BertIntermediate(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# ---- mmpretrain.models.multimodal.blip.language_model.find_pruneable_heads_and_indices ----
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

# ---- mmpretrain.models.multimodal.blip.language_model.prune_linear_layer ----
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

# ---- mmpretrain.models.multimodal.blip.language_model.BertAttention ----
class BertAttention(nn.Module):

    def __init__(self, config, is_cross_attention=False, layer_num=-1):
        super().__init__()
        is_nlvr = is_cross_attention and getattr(config, 'nlvr', False)
        if is_nlvr:
            self.self0 = BertSelfAttention(config, is_nlvr)
            self.self1 = BertSelfAttention(config, is_nlvr)
        else:
            self.self = BertSelfAttention(config, is_cross_attention)
        self.output = BertSelfOutput(
            config,
            twin=is_nlvr,
            merge=(is_nlvr and layer_num >= 6),
        )
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads,
            self.self.num_attention_heads,
            self.self.attention_head_size,
            self.pruned_heads,
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(
            heads)
        self.self.all_head_size = (
            self.self.attention_head_size * self.self.num_attention_heads)
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        if type(encoder_hidden_states) == list:
            self_outputs0 = self.self0(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states[0],
                encoder_attention_mask[0],
                past_key_value,
                output_attentions,
            )
            self_outputs1 = self.self1(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states[1],
                encoder_attention_mask[1],
                past_key_value,
                output_attentions,
            )
            attention_output = self.output(
                [self_outputs0[0], self_outputs1[0]], hidden_states)

            outputs = (attention_output, ) + self_outputs0[
                1:]  # add attentions if we output them
        else:
            self_outputs = self.self(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions,
            )
            attention_output = self.output(self_outputs[0], hidden_states)
            outputs = (attention_output,
                       ) + self_outputs[1:]  # add attentions if we output them
        return outputs

# ---- mmpretrain.models.multimodal.blip.language_model.apply_chunking_to_forward ----
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

# ---- BertLayer (target) ----
class BertLayer(nn.Module):

    def __init__(self, config, layer_num):
        super().__init__()
        self.config = config
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = BertAttention(config)
        self.layer_num = layer_num
        if self.config.add_cross_attention:
            self.crossattention = BertAttention(
                config, is_cross_attention=self.config.add_cross_attention)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        mode=None,
    ):

        if mode == 'tagging':

            assert encoder_hidden_states is not None, \
                '''encoder_hidden_states must be given
                for cross-attention layers'''

            cross_attention_outputs = self.crossattention(
                hidden_states,
                attention_mask,
                head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                output_attentions=output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = cross_attention_outputs[
                1:-1]  # add cross attentions if we output attention weights

            present_key_value = cross_attention_outputs[-1]

        else:
            # decoder uni-directional self-attention
            # cached key/values tuple is at positions 1,2
            self_attn_past_key_value = \
                (past_key_value[:2]
                    if past_key_value is not None else None)
            self_attention_outputs = self.attention(
                hidden_states,
                attention_mask,
                head_mask,
                output_attentions=output_attentions,
                past_key_value=self_attn_past_key_value,
            )
            attention_output = self_attention_outputs[0]

            outputs = self_attention_outputs[1:-1]
            present_key_value = self_attention_outputs[-1]

            if mode == 'multimodal':
                assert encoder_hidden_states is not None, \
                    '''encoder_hidden_states must be
                    given for cross-attention layers'''

                cross_attention_outputs = self.crossattention(
                    attention_output,
                    attention_mask,
                    head_mask,
                    encoder_hidden_states,
                    encoder_attention_mask,
                    output_attentions=output_attentions,
                )
                attention_output = cross_attention_outputs[0]
                outputs = outputs + cross_attention_outputs[
                    1:
                    -1]  # add cross attentions if we output attention weights
        layer_output = apply_chunking_to_forward(self.feed_forward_chunk,
                                                 self.chunk_size_feed_forward,
                                                 self.seq_len_dim,
                                                 attention_output)
        outputs = (layer_output, ) + outputs

        outputs = outputs + (present_key_value, )

        return outputs

    def feed_forward_chunk(self, attention_output):
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output
