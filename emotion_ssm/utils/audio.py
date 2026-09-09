"""Length-aware pooling independent of the acoustic model's mask policy."""
import torch


def minimum_waveform_length(model):
    length = 1
    for kernel, stride in reversed(list(zip(getattr(model.config, "conv_kernel", [1]),
                                           getattr(model.config, "conv_stride", [1])))):
        length = (length-1) * stride + kernel
    return length


def audio_hidden(model, waveform, lengths=None):
    if lengths is None:
        lengths = torch.full((len(waveform),), waveform.shape[1],
                             device=waveform.device, dtype=torch.long)
    lengths = lengths.to(device=waveform.device, dtype=torch.long)
    if lengths.shape != (len(waveform),) or (lengths < 0).any() or (lengths > waveform.shape[1]).any():
        raise ValueError("Audio lengths must be within the supplied waveform")
    minimum = minimum_waveform_length(model)
    if not (lengths >= minimum).any():
        return waveform.new_zeros(len(waveform), 1, model.config.hidden_size), torch.zeros(
            len(waveform), 1, dtype=torch.bool, device=waveform.device)
    # Group-normalized Wav2Vec2 sees each valid waveform separately: padding
    # otherwise changes the convolution's normalization even with masked pooling.
    if getattr(model.config, "feat_extract_norm", "group") == "group":
        outputs = []
        for row, length in zip(waveform, lengths):
            if int(length) < minimum:
                outputs.append(waveform.new_zeros((1, model.config.hidden_size)))
            else:
                outputs.append(model(row[None, :int(length)]).last_hidden_state[0])
        hidden = torch.nn.utils.rnn.pad_sequence(outputs, batch_first=True)
        output_lengths = torch.tensor([len(x) if int(n) >= minimum else 0
                                       for x, n in zip(outputs, lengths)], device=waveform.device)
    else:
        mask = torch.arange(waveform.shape[1], device=waveform.device)[None] < lengths[:, None]
        usable = lengths >= minimum
        selected = model(waveform[usable], attention_mask=mask[usable].long()).last_hidden_state
        hidden = selected.new_zeros(len(waveform), selected.shape[1], selected.shape[2])
        hidden[usable] = selected
        output_lengths = model._get_feat_extract_output_lengths(lengths).clamp_min(0)
    mask = torch.arange(hidden.shape[1], device=waveform.device)[None] < output_lengths[:, None]
    return hidden, mask


def pooled_audio(model, waveform, lengths=None):
    hidden, mask = audio_hidden(model, waveform, lengths)
    return (hidden * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp_min(1)
