"""Identical sufficient statistics for train validation, DDP and evaluation."""
import torch
import torch.distributed as dist


class ReconstructionTotals:
    names = ("expression", "jaw", "neck", "velocity", "boundary_velocity")

    def __init__(self):
        self.square_error = dict.fromkeys(self.names, 0.)
        self.elements = dict.fromkeys(self.names, 0)
        self.chunks = 0

    def update(self, generated, target, mask=None, previous=None):
        generated, target = generated.detach().double(), target.detach().double()
        if mask is None:
            mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        groups = {"expression": (generated[..., :50], target[..., :50], mask),
                  "jaw": (generated[..., 50:53], target[..., 50:53], mask),
                  "neck": (generated[..., 53:56], target[..., 53:56], mask),
                  "velocity": (generated[:, 1:] - generated[:, :-1],
                               target[:, 1:] - target[:, :-1], mask[:, 1:] & mask[:, :-1])}
        if previous is not None:
            pg, pt = previous[:2]
            boundary_mask = mask[:, :1]
            if len(previous) == 3:
                boundary_mask = boundary_mask & previous[2][:, None].bool()
            groups["boundary_velocity"] = (generated[:, :1] - pg.double()[:, None],
                                            target[:, :1] - pt.double()[:, None], boundary_mask)
        for name, (prediction, truth, valid) in groups.items():
            self.square_error[name] += float(torch.where(valid[..., None], (prediction-truth).square(), 0.).sum())
            self.elements[name] += int(valid.sum()) * prediction.shape[-1]
        self.chunks += int(mask.any(1).sum())

    def merge(self, other):
        for name in self.names:
            self.square_error[name] += other.square_error[name]
            self.elements[name] += other.elements[name]
        self.chunks += other.chunks
        return self

    def distributed_sum(self, device):
        if dist.is_available() and dist.is_initialized():
            values = [self.square_error[n] for n in self.names] + [self.elements[n] for n in self.names] + [self.chunks]
            values = torch.tensor(values, dtype=torch.float64, device=device)
            dist.all_reduce(values)
            k = len(self.names)
            for i, n in enumerate(self.names):
                self.square_error[n], self.elements[n] = float(values[i]), int(values[k+i])
            self.chunks = int(values[-1])
        return self

    def metrics(self):
        result = {f"{n}_mse": self.square_error[n] / max(self.elements[n], 1) for n in self.names}
        result["generation_total"] = sum(result[f"{n}_mse"] for n in self.names[:4])
        result["evaluated_chunks"] = self.chunks
        result["valid_frames"] = self.elements["expression"] // 50
        return result


def reconstruction_loss(generated, target, mask=None):
    if mask is None:
        mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
    def mse(x, y, valid):
        return ((x-y).square() * valid[..., None]).sum() / (valid.sum() * x.shape[-1]).clamp_min(1)
    values = {"expression": mse(generated[..., :50], target[..., :50], mask),
              "jaw": mse(generated[..., 50:53], target[..., 50:53], mask),
              "neck": mse(generated[..., 53:56], target[..., 53:56], mask),
              "velocity": mse(generated[:, 1:]-generated[:, :-1], target[:, 1:]-target[:, :-1], mask[:, 1:] & mask[:, :-1])}
    values["total"] = sum(values.values())
    return values
