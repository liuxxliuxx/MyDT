"""Frozen feature extraction shared by offline preparation and streaming."""
from __future__ import annotations

import torch
from torch import nn

from emotion_ssm.data.protocol import role_text, visible_words
from emotion_ssm.utils.audio import pooled_audio, minimum_waveform_length


def auto_config(values):
    from transformers import AutoConfig
    values = dict(values)
    model_type = values.pop("model_type")
    return AutoConfig.for_model(model_type, **values)


class FrozenFeatures(nn.Module):
    def __init__(self, source, construction=None, local_files_only=False, max_tokens=256):
        super().__init__()
        from transformers import AutoModel, AutoTokenizer, PreTrainedTokenizerFast
        self.source = dict(source)
        self.max_tokens = max_tokens
        if construction is None:
            self.audio = AutoModel.from_pretrained(source["audio_model"], local_files_only=local_files_only,
                                                  revision=source.get("audio_revision") or None)
            self.text = AutoModel.from_pretrained(source["text_model"], local_files_only=local_files_only,
                                                 revision=source.get("text_revision") or None)
            self.tokenizer = AutoTokenizer.from_pretrained(source["text_model"], use_fast=True,
                local_files_only=local_files_only, revision=source.get("text_revision") or None)
            for name, model in (("audio", self.audio), ("text", self.text)):
                from emotion_ssm.data.protocol import model_revision
                resolved = model_revision(source[name+"_model"], model.config)
                if resolved:
                    if source.get(name+"_revision") and source[name+"_revision"] != resolved:
                        raise ValueError(f"{name} model changed since the declared feature revision")
                    self.source[name+"_revision"] = resolved
        else:
            from tokenizers import Tokenizer
            self.audio = AutoModel.from_config(auto_config(construction["audio"]))
            self.text = AutoModel.from_config(auto_config(construction["text"]))
            self.tokenizer = PreTrainedTokenizerFast(
                tokenizer_object=Tokenizer.from_str(construction["tokenizer"]),
                **construction["special_tokens"])
        self.tokenizer.truncation_side = "left"
        if self.audio.config.hidden_size != source["audio_dim"] or self.text.config.hidden_size != source["text_dim"]:
            raise ValueError("Feature dimensions disagree with the declared extraction models")
        self.requires_grad_(False)
        self.eval()

    def construction(self):
        return {"audio": self.audio.config.to_dict(), "text": self.text.config.to_dict(),
                "tokenizer": self.tokenizer.backend_tokenizer.to_str(),
                "special_tokens": self.tokenizer.special_tokens_map}

    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode_text(self, text):
        if not text.strip():
            return next(self.text.parameters()).new_zeros(1, self.text.config.hidden_size)
        device = next(self.text.parameters()).device
        inputs = self.tokenizer([text], return_tensors="pt", truncation=True,
                                max_length=self.max_tokens, padding=True).to(device)
        hidden = self.text(**inputs).last_hidden_state
        mask = inputs["attention_mask"]
        return (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)

    @torch.no_grad()
    def forward(self, audio, words, now, since, role, speech_active=True, audio_length=None):
        # Input is the newly completed block, never a future or whole-clip slice.
        length = audio.shape[-1] if audio_length is None else int(audio_length)
        if length < 0 or length > audio.shape[-1]:
            raise ValueError("Audio valid length exceeds the supplied block")
        speech_active = bool(speech_active and length >= minimum_waveform_length(self.audio))
        valid_wave = audio[:, :length]
        if length > 0 and speech_active:
            valid_wave = (valid_wave - valid_wave.mean(-1, keepdim=True)) / valid_wave.std(-1, keepdim=True, unbiased=False).clamp_min(1e-6)
            audio_value = pooled_audio(self.audio, valid_wave)
        else:
            audio_value = audio.new_zeros(1, self.source["audio_dim"])
        history = visible_words(words, now)
        new = [w for w in visible_words(words, now, since) if str(w["role"]) == str(role)]
        text_value = self.encode_text(role_text(history))
        return {"audio": audio_value, "text": text_value,
                "event_text": self.encode_text(role_text(new)),
                "modality_mask": torch.tensor([[bool(speech_active and length), False, bool(history)]], device=audio.device),
                "event_present": torch.tensor([bool(new)], device=audio.device)}
