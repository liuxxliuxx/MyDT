"""Identical sufficient statistics for train validation, DDP and evaluation."""
import torch
import torch.distributed as dist


class ExpressionDecomposition:
    """Per-conversation temporal moments; no target-derived normalization in inference."""

    def __init__(self, device):
        self.values = torch.zeros(5, 50, dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    @torch.no_grad()
    def update(self, prediction, target, mask):
        p = torch.where(mask[..., None], prediction[..., :50].detach().double(), 0.)
        t = torch.where(mask[..., None], target[..., :50].detach().double(), 0.)
        self.values += torch.stack([p.sum((0, 1)), t.sum((0, 1)), p.square().sum((0, 1)),
                                   t.square().sum((0, 1)), (p - t).square().sum((0, 1))])
        self.count += mask.sum()

    def metrics(self):
        means = self.values / self.count.clamp_min(1)
        bias = (means[0] - means[1]).square().mean()
        return dict(expression_bias_mse=float(bias), expression_centered_mse=float((means[4].mean() - bias).clamp_min(0)),
                    expression_pred_variance=float((means[2] - means[0].square()).mean().clamp_min(0)),
                    expression_target_variance=float((means[3] - means[1].square()).mean().clamp_min(0)))


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


class DeferredReconstructionTotals(ReconstructionTotals):
    """Keep the original FP64 reductions/order, but read the totals only once.

    These tensors are detached diagnostics. They never enter the loss graph or
    change its precision. Each chunk is still added in chronological order.
    """

    def __init__(self, device):
        super().__init__()
        self._values = torch.zeros(11, dtype=torch.float64, device=device)
        self._materialized = False

    @torch.no_grad()
    def update(self, generated, target, mask=None, previous=None):
        if self._materialized:
            raise RuntimeError("Cannot append chunks after deferred totals were read")
        generated, target = generated.detach().double(), target.detach().double()
        if mask is None:
            mask = torch.ones(target.shape[:2], dtype=torch.bool, device=target.device)
        groups = [(generated[..., :50], target[..., :50], mask),
                  (generated[..., 50:53], target[..., 50:53], mask),
                  (generated[..., 53:56], target[..., 53:56], mask),
                  (generated[:, 1:] - generated[:, :-1], target[:, 1:] - target[:, :-1],
                   mask[:, 1:] & mask[:, :-1])]
        if previous is not None:
            pg, pt = previous[:2]
            boundary_mask = mask[:, :1]
            if len(previous) == 3:
                boundary_mask = boundary_mask & previous[2][:, None].bool()
            groups.append((generated[:, :1] - pg.double()[:, None],
                           target[:, :1] - pt.double()[:, None], boundary_mask))
        sums, counts = [], []
        for prediction, truth, valid in groups:
            sums.append(torch.where(valid[..., None], (prediction-truth).square(), 0.).sum())
            counts.append(valid.sum() * prediction.shape[-1])
        if previous is None:
            sums.append(self._values.new_zeros(()))
            counts.append(self._values.new_zeros(()))
        row = torch.stack(sums + counts + [mask.any(1).sum()]).to(torch.float64)
        self._values.add_(row)

    def distributed_sum(self, device):
        if self._materialized:
            raise RuntimeError("Deferred totals can only be reduced once")
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(self._values)
        values = self._values.cpu().tolist()
        for index, name in enumerate(self.names):
            self.square_error[name] = values[index]
            self.elements[name] = int(values[index + len(self.names)])
        self.chunks = int(values[-1])
        self._materialized = True

    def metrics(self):
        if not self._materialized:
            raise RuntimeError("Call distributed_sum before reading deferred metrics")
        return super().metrics()


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
