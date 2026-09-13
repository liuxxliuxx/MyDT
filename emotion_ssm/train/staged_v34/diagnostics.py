"""Independent task-gradient diagnostics for the v3.4 objective."""
import torch
from emotion_ssm.utils.checkpoint import capture_rng_state,restore_rng_state
from emotion_ssm.train.staged_v33.optimization import reduced,world
from .data import QuerySampler
from .objectives import loss,supports


def gradient_diagnostic(observer,core,bank,settings,budget,gold_budget,rank,size,seed,online=False):
    rng=capture_rng_state()
    try:
        sampler=QuerySampler(bank,seed);v,g=sampler.sample(budget,gold_budget,rank,size)
        support=supports(sampler,next(core.parameters()).device)
        parameters=[p for module in (observer,core) for p in module.parameters() if p.requires_grad]
        gradients={}
        for name in ('vector','ce','vad','intensity','anchor','semantic_keep','neutral_margin'):
            restore_rng_state(rng)
            _,terms,_=loss(core,observer,bank,v,g,settings,support,online)
            values=torch.autograd.grad(terms[name],parameters,allow_unused=True,retain_graph=False)
            flat=torch.cat([torch.zeros_like(p).reshape(-1) if grad is None else grad.detach().reshape(-1)
                            for p,grad in zip(parameters,values)])
            gradients[name]=reduced(flat)/world()
        vector=gradients['vector'];result={}
        for name,grad in gradients.items():
            norm=grad.norm();vn=vector.norm()
            result[name]=dict(norm=float(norm),ratio_to_vector=float(norm/vn.clamp_min(1e-12)),
                cosine_to_vector=float(torch.dot(grad,vector)/(norm*vn).clamp_min(1e-12)))
        return dict(seed=seed,independent_rng=True,graph_protocol='fresh_forward_per_task_v1',raw_task_gradients=result)
    finally:restore_rng_state(rng)
