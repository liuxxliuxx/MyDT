import torch
import torch.nn.functional as F
import numpy as np
from transformers import Wav2Vec2Model, Wav2Vec2ForCTC
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from typing import Optional, Tuple

_CONFIG_FOR_DOC = "Wav2Vec2Config"
_HIDDEN_STATES_START_POSITION = 2


# the implementation of Wav2Vec2Model is borrowed from https://huggingface.co/transformers/_modules/transformers/models/wav2vec2/modeling_wav2vec2.html#Wav2Vec2Model
# initialize our encoder with the pre-trained wav2vec 2.0 weights.
def _compute_mask_indices(
        shape: Tuple[int, int],
        mask_prob: float,
        mask_length: int,
        attention_mask: Optional[torch.Tensor] = None,
        min_masks: int = 0,
) -> np.ndarray:
    bsz, all_sz = shape
    mask = np.full((bsz, all_sz), False)

    all_num_mask = int(
        mask_prob * all_sz / float(mask_length)
        + np.random.rand()
    )
    all_num_mask = max(min_masks, all_num_mask)
    mask_idcs = []
    padding_mask = attention_mask.ne(1) if attention_mask is not None else None
    for i in range(bsz):
        if padding_mask is not None:
            sz = all_sz - padding_mask[i].long().sum().item()
            num_mask = int(
                mask_prob * sz / float(mask_length)
                + np.random.rand()
            )
            num_mask = max(min_masks, num_mask)
        else:
            sz = all_sz
            num_mask = all_num_mask

        if num_mask == 0 or sz <= 0:
            mask_idcs.append(np.empty(0, dtype=np.int64))
            continue

        lengths = np.full(num_mask, mask_length)

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        min_len = min(lengths)
        if sz - min_len <= num_mask:
            min_len = sz - num_mask - 1

        mask_idc = np.random.choice(sz - min_len, num_mask, replace=False)
        mask_idc = np.asarray([mask_idc[j] + offset for j in range(len(mask_idc)) for offset in range(lengths[j])])
        mask_idcs.append(np.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = np.random.choice(mask_idc, min_len, replace=False)
        mask[i, mask_idc] = True
    return mask


def _budget_mask_indices(shape, mask_prob, mask_length, attention_mask=None):
    """Versioned short-window policy: expected masked positions = prob * valid.

    Stochastic rounding allows zero masks. Spans shrink to the remaining budget,
    never overlap or touch padding. Each row keeps its own valid-length budget.
    """
    if not 0 <= mask_prob <= 1 or mask_length < 1:
        raise ValueError("Invalid time-mask probability or span length")
    valid = (np.ones(shape, dtype=bool) if attention_mask is None else
             attention_mask.detach().cpu().numpy().astype(bool))
    if valid.shape != shape:
        raise ValueError("Time mask must use the interpolated feature grid")
    mask = np.zeros(shape, dtype=bool)
    for row in range(shape[0]):
        positions = np.flatnonzero(valid[row])
        budget = min(len(positions), int(mask_prob * len(positions) + np.random.rand()))
        available = valid[row].copy()
        while budget:
            start = int(np.random.choice(np.flatnonzero(available)))
            for index in range(start, min(shape[1], start + mask_length)):
                if not available[index]:
                    break
                mask[row, index], available[index] = True, False
                budget -= 1
                if not budget:
                    break
    return mask


# linear interpolation layer
def linear_interpolation(features, input_fps, output_fps, output_len=None):
    features = features.transpose(1, 2)
    seq_len = features.shape[2] / float(input_fps)
    if output_len is None:
        output_len = int(seq_len * output_fps)
    output_features = F.interpolate(features, size=output_len, align_corners=True, mode='linear')
    return output_features.transpose(1, 2)


class Wav2Vec2Model(Wav2Vec2Model):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_values,
            dataset="vocaset",
            attention_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            frame_num=None
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        hidden_states = self.feature_extractor(input_values)
        hidden_states = hidden_states.transpose(1, 2)
        convolution_length = hidden_states.shape[1]

        if dataset == "vocaset":
            hidden_states = linear_interpolation(hidden_states, 50, 30, output_len=frame_num)

        if attention_mask is not None:
            output_lengths = self._get_feat_extract_output_lengths(attention_mask.sum(-1))
            output_lengths = torch.ceil(output_lengths.clamp_min(0).double() *
                                        hidden_states.shape[1] / convolution_length).long()
            attention_mask = (torch.arange(hidden_states.shape[1], device=hidden_states.device)[None]
                              < output_lengths[:, None])

        hidden_states = self.feature_projection(hidden_states)[0]

        if self.config.apply_spec_augment and self.training:
            batch_size, sequence_length, hidden_size = hidden_states.size()
            if self.config.mask_time_prob > 0:
                policy = getattr(self.config, "avatar_mask_policy", "legacy_min_two")
                if policy == "valid_budget_v1":
                    mask_time_indices = _budget_mask_indices((batch_size, sequence_length),
                        self.config.mask_time_prob, self.config.mask_time_length, attention_mask)
                elif policy == "legacy_min_two":
                    mask_time_indices = _compute_mask_indices((batch_size, sequence_length),
                        self.config.mask_time_prob, self.config.mask_time_length,
                        attention_mask=attention_mask, min_masks=2)
                else:
                    raise ValueError("Unknown Avatar time-mask protocol: " + policy)
                # CPU mask statistics introduce no extra device synchronization
                # for the unpadded generator inputs used in streaming training.
                if policy == "valid_budget_v1" and attention_mask is None:
                    old = getattr(self, "_avatar_mask_totals", [0, 0, 0, 0])
                    current = mask_time_indices[:, -25:]
                    self._avatar_mask_totals = [old[0] + int(mask_time_indices.sum()), old[1] + mask_time_indices.size,
                                               old[2] + int(current.sum()), old[3] + current.size]
                hidden_states[torch.from_numpy(mask_time_indices).to(hidden_states.device)] = self.masked_spec_embed.to(hidden_states.dtype)
            if self.config.mask_feature_prob > 0:
                mask_feature_indices = _compute_mask_indices(
                    (batch_size, hidden_size),
                    self.config.mask_feature_prob,
                    self.config.mask_feature_length,
                )
                mask_feature_indices = torch.from_numpy(mask_feature_indices).to(hidden_states.device)
                hidden_states[mask_feature_indices[:, None].expand(-1, sequence_length, -1)] = 0
        encoder_outputs = self.encoder(
            hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = encoder_outputs[0]
        if not return_dict:
            return (hidden_states,) + encoder_outputs[1:]

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )


class Wav2Vec2ForCTC(Wav2Vec2ForCTC):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_values,
            attention_mask=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            frame_num=None
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        hidden_states = outputs[0]
        logits = self.lm_head(self.dropout(hidden_states))

        loss = None
        if labels is not None:
            loss = self._compute_loss(logits, labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutput(
            loss=loss, logits=logits, hidden_states=hidden_states, attentions=outputs.attentions
        )
