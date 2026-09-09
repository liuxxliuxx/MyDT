"""Offline CTC alignment of supplied text; never performs transcription."""
from __future__ import annotations

import re
import torch


def ctc_viterbi(log_probs, tokens, blank=0):
    """Standard blank-expanded CTC trellis, including repeated-token rules.

    Returns inclusive/exclusive frame intervals for every transcript token.
    An impossible path raises, rather than fabricating timestamps.
    """
    scores = log_probs.detach().float().cpu()
    tokens = list(tokens)
    if not tokens:
        return []
    if scores.ndim != 2 or not len(scores) or not torch.isfinite(scores).all():
        raise ValueError("CTC emissions must be finite [time, vocabulary]")
    if blank in tokens or min([blank]+tokens) < 0 or max([blank]+tokens) >= scores.shape[1]:
        raise ValueError("Invalid transcript token or blank ID")
    expanded = [blank]
    for token in tokens:
        expanded.extend([token, blank])
    labels = torch.tensor(expanded)
    steps, width = scores.shape[0], len(labels)
    previous = torch.full((width,), -float("inf"))
    previous[0] = scores[0, blank]
    previous[1] = scores[0, labels[1]]
    paths = torch.zeros((steps, width), dtype=torch.int8)
    for time in range(1, steps):
        stay = previous
        advance = torch.cat([previous.new_full((1,), -float("inf")), previous[:-1]])
        skip = torch.cat([previous.new_full((2,), -float("inf")), previous[:-2]])
        allowed = torch.zeros(width, dtype=torch.bool)
        allowed[2:] = (labels[2:] != blank) & (labels[2:] != labels[:-2])
        skip = skip.masked_fill(~allowed, -float("inf"))
        best, choice = torch.stack([stay, advance, skip]).max(0)
        previous = best + scores[time, labels]
        paths[time] = choice.to(torch.int8)
    state = width - 1 if previous[-1] >= previous[-2] else width - 2
    if not torch.isfinite(previous[state]):
        raise ValueError("Transcript has no valid alignment to this audio")
    assignments = [[] for _ in tokens]
    for time in range(steps - 1, -1, -1):
        if state % 2:
            assignments[state // 2].append(time)
        state -= int(paths[time, state])
    if any(not frames for frames in assignments):
        raise ValueError("Incomplete CTC alignment")
    return [(min(frames), max(frames) + 1,
             float(scores[frames, token].exp().mean()))
            for frames, token in zip(assignments, tokens)]


class TranscriptAligner:
    def __init__(self, model_name="facebook/wav2vec2-base-960h", device="cpu", local_files_only=False):
        from transformers import AutoProcessor, AutoModelForCTC
        self.processor = AutoProcessor.from_pretrained(model_name, local_files_only=local_files_only)
        self.model = AutoModelForCTC.from_pretrained(model_name, local_files_only=local_files_only).to(device).eval()
        self.device, self.model_name = device, model_name

    @torch.no_grad()
    def align(self, waveform, transcript, role, minimum_score=.05):
        if any(char.isalnum() and char not in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" for char in transcript.upper()):
            raise ValueError("Unsupported transcript symbols; text is missing, not automatically rewritten")
        words = re.findall(r"[A-Z]+(?:'[A-Z]+)?", transcript.upper())
        if not words:
            return []
        vocab = self.processor.tokenizer.get_vocab()
        separator = self.processor.tokenizer.word_delimiter_token
        characters = list(separator.join(words))
        if any(char not in vocab for char in characters):
            raise ValueError("Transcript contains unsupported characters")
        inputs = self.processor(waveform, sampling_rate=16000, return_tensors="pt")
        scores = self.model(inputs.input_values.to(self.device)).logits[0].log_softmax(-1)
        intervals = ctc_viterbi(scores, [vocab[c] for c in characters], self.model.config.pad_token_id)
        frame_seconds = len(waveform) / 16000 / len(scores)
        result, offset = [], 0
        for index, word in enumerate(words):
            aligned = intervals[offset:offset+len(word)]
            confidence = sum(x[2] for x in aligned) / len(aligned)
            if confidence < minimum_score:
                raise ValueError("Low-confidence transcript alignment; mask the clip text")
            start, end = aligned[0][0] * frame_seconds, aligned[-1][1] * frame_seconds
            result.append({"id": f"{role}:{index}", "role": role, "text": word.lower(),
                           "start": start, "end": end, "available_at": end, "confidence": confidence})
            offset += len(word) + 1
        return result
