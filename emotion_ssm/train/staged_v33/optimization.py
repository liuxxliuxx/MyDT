"""Global-count objectives and persistent, disjoint AdamW groups."""
import math
import torch
import torch.distributed as dist

from emotion_ssm.train.staged_dynamics_support import LOCKED_CORE, UPDATE_PREFIXES, HEAD_PREFIXES


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def reduced(value):
    result = value.detach().clone()
    if dist.is_initialized():
        dist.all_reduce(result)
    return result


def global_mean(numerators, denominators, support=None, require_complete=False):
    """Mean domains within horizons, then supported horizons; W corrects rank averaging."""
    counts = reduced(denominators.double())
    present = counts > 0
    support = present if support is None else support.to(counts.device).bool()
    if require_complete and bool((support & ~present).any()):
        raise ValueError('Effective optimizer batch does not cover declared domain/horizon cells')
    active = support & present
    per_h = active.sum(0)
    nh = (per_h > 0).sum().clamp_min(1)
    weights = active / per_h.clamp_min(1)[None] / nh / counts.clamp_min(1)
    local = (numerators * weights.to(numerators.dtype)).sum() * world()
    sums = reduced(numerators.double())
    value = (sums * weights).sum().to(local.dtype)
    return local + (value-local.detach()), dict(sums=sums.cpu().tolist(),
        counts=counts.cpu().tolist(), supported=support.cpu().tolist(), value=float(value.detach()))


def synchronize(parameters):
    """Average gradients with a stable bucket; globally unused parameters stay None."""
    if not parameters:
        return
    used = torch.tensor([p.grad is not None for p in parameters], device=parameters[0].device,
                        dtype=torch.int32)
    if dist.is_initialized():
        dist.all_reduce(used, op=dist.ReduceOp.MAX)
    selected = list(zip(parameters,used.cpu().tolist()))
    values = [torch.zeros_like(p).reshape(-1) if p.grad is None else p.grad.reshape(-1)
              for p,active in selected if active]
    if not values:
        return
    flat = torch.cat(values)
    if dist.is_initialized():
        dist.all_reduce(flat)
        flat /= world()
    if not bool(torch.isfinite(flat).all()):
        raise FloatingPointError('Non-finite globally reduced gradient')
    offset = 0
    for p,active in selected:
        if not active:
            p.grad = None
            continue
        p.grad = flat[offset:offset+p.numel()].view_as(p)
        offset += p.numel()


class PersistentOptimizers:
    def __init__(self, observer, core, settings):
        self.settings = settings
        self.named = {'update': [], 'flow': []}
        observer.requires_grad_(False); core.requires_grad_(False)
        for name,p in observer.named_parameters():
            if name.startswith(HEAD_PREFIXES):
                self.named['update'].append(('observer.'+name,p))
        for name,p in core.named_parameters():
            if name not in LOCKED_CORE:
                self.named['update' if name.startswith(UPDATE_PREFIXES) else 'flow'].append(('state.'+name,p))
        identities = [id(p) for group in self.named.values() for _,p in group]
        if len(set(identities)) != len(identities):
            raise ValueError('Optimizer parameter groups overlap')
        self.optimizers = {name: torch.optim.AdamW([p for _,p in pairs], lr=1e-4,
            weight_decay=settings.get('weight_decay',.001)) for name,pairs in self.named.items()}
        self.updates = dict(update=0, flow=0)
        self.active = []

    def phase(self, phase):
        self.active = (['update'] if phase=='calibration' else ['update','flow'] if phase=='joint' else ['flow'])
        rates = {'update': self.settings['calibration_lr'] if phase=='calibration' else self.settings['joint_lr'],
                 'flow': self.settings['prediction_lr'] if phase=='fixed' else
                         self.settings['joint_lr'] if phase=='joint' else self.settings['joint_prediction_lr']}
        for name,pairs in self.named.items():
            for _,p in pairs:
                p.requires_grad_(name in self.active)
                p.grad = None
            factor = 1.
            if self.settings.get('schedule','constant')=='cosine':
                fraction = min(1., self.updates[name]/max(1,self.settings['schedule_updates'][name]))
                floor = self.settings.get('min_lr_ratio',.1)
                factor = floor+(1-floor)*.5*(1+math.cos(math.pi*fraction))
            for group in self.optimizers[name].param_groups:
                group['lr'] = rates[name]*factor

    def parameters(self):
        return [p for name in self.active for _,p in self.named[name]]

    def zero_grad(self):
        for optimizer in self.optimizers.values():
            optimizer.zero_grad(set_to_none=True)

    def step(self):
        parameters = self.parameters()
        synchronize(parameters)
        norm = torch.nn.utils.clip_grad_norm_(parameters, self.settings.get('clip_grad',5.))
        for name in self.active:
            if any(p.grad is not None for _,p in self.named[name]):
                self.optimizers[name].step()
                self.updates[name] += 1
        return float(norm)

    def state_dict(self):
        return dict(names={k:[n for n,_ in v] for k,v in self.named.items()},
                    optimizers={k:o.state_dict() for k,o in self.optimizers.items()}, updates=self.updates.copy())

    def load_state_dict(self, value):
        if value['names'] != {k:[n for n,_ in v] for k,v in self.named.items()}:
            raise ValueError('Persistent optimizer parameter mapping changed')
        for name, optimizer in self.optimizers.items():
            optimizer.load_state_dict(value['optimizers'][name])
        self.updates = value['updates'].copy()
