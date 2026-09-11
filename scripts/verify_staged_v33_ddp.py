"""CUDA/NCCL audit of the actual model loss, gradients, updates and metrics."""
import argparse
import contextlib
import copy
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
import torch
import torch.distributed as dist


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True);args=parser.parse_args()
    rank=int(os.environ['LOCAL_RANK']);torch.cuda.set_device(rank);device=torch.device('cuda',rank)
    dist.init_process_group('nccl');torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    root=Path(args.output);root.mkdir(parents=True,exist_ok=True)
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tests'))
    from test_staged_v33 import case
    from emotion_ssm.models.token_observer import TokenObserver
    from emotion_ssm.models.state_core import UnifiedEmotionStateCore
    from emotion_ssm.train.staged_v33.optimization import PersistentOptimizers
    from emotion_ssm.train.staged_v33.data import QuerySampler
    from emotion_ssm.train.staged_v33.objectives import loss,supports
    from emotion_ssm.train.staged_v33.baselines import fit
    from emotion_ssm.train.staged_v33.evaluation import evaluate
    from emotion_ssm.train.staged_v33.diagnostics import gradient_diagnostic
    if rank==0:
        fixture=root/'fixture';fixture.mkdir(exist_ok=True)
        _,o,t,c,_,_,bank,settings=case(fixture)
        bank['domain']=torch.tensor([0,0,1,1,1,1,2,2])
        torch.save(dict(observer=o.state_dict(),core=c.state_dict(),observer_config=o.construction(),
            core_config=c.get_config(),bank=bank,settings=settings),root/'case.pt')
    dist.barrier();data=torch.load(root/'case.pt',map_location='cpu',weights_only=False)
    results=[]
    for affine in (False,True):
        for phase in ('fixed','joint'):
            for execution in ('reference','compiled'):
                def construct():
                    observer=TokenObserver(data['observer_config']).to(device).eval();observer.load_state_dict(data['observer'])
                    config=dict(data['core_config'])
                    if affine:config.update(flow_kind='adaptive_affine_dyadic_v2',affine_max_offset=.5)
                    core=UnifiedEmotionStateCore.from_config(config).to(device);core.load_state_dict(data['core'],strict=not affine)
                    core.configure_execution(execution)
                    optimizer=PersistentOptimizers(observer,core,data['settings']);optimizer.phase(phase)
                    return observer,core,optimizer
                observer,core,optimizer=construct();bank=data['bank'];settings=data['settings']
                sampler=QuerySampler(bank,998);v,g=sampler.sample(32,32,rank,2)
                total,terms,stats=loss(core,observer,bank,v,g,settings,supports(sampler,device),phase=='joint')
                total.backward();optimizer.step()
                actual={n:p.detach().clone() for n,p in list(observer.named_parameters())+[('core.'+n,p) for n,p in core.named_parameters()]}
                if rank==0:
                    serial_observer,serial_core,serial_optimizer=construct();serial_sampler=QuerySampler(bank,998)
                    v,g=serial_sampler.sample(32,32)
                    with patch('emotion_ssm.train.staged_v33.optimization.reduced',lambda x:x.detach().clone()),\
                         patch('emotion_ssm.train.staged_v33.optimization.world',lambda:1),\
                         patch('emotion_ssm.train.staged_v33.optimization.synchronize',lambda p:None):
                        expected,_,_=loss(serial_core,serial_observer,bank,v,g,settings,supports(serial_sampler,device),phase=='joint')
                        expected.backward();serial_optimizer.step()
                    torch.testing.assert_close(total.detach(),expected.detach(),rtol=2e-5,atol=2e-6)
                    maximum=0.
                    for name,p in list(serial_observer.named_parameters())+[('core.'+n,p) for n,p in serial_core.named_parameters()]:
                        maximum=max(maximum,float((actual[name]-p).abs().max()))
                        torch.testing.assert_close(actual[name],p,rtol=2e-5,atol=3e-6)
                    result=dict(affine=affine,phase=phase,execution=execution,loss=float(total),max_parameter_delta=maximum)
                    results.append(result);print(json.dumps(result),flush=True)
                dist.barrier()
                saved_rng=torch.get_rng_state().clone()
                saved_grads=[None if p.grad is None else p.grad.clone() for p in optimizer.parameters()]
                diagnostic=gradient_diagnostic(observer,core,bank,settings,32,32,rank,2,771,phase=='joint')
                assert torch.equal(saved_rng,torch.get_rng_state())
                for before,p in zip(saved_grads,optimizer.parameters()):
                    if before is None:assert p.grad is None
                    else:torch.testing.assert_close(before,p.grad,rtol=0,atol=0)
                assert all(torch.isfinite(torch.tensor(v['norm'])) for v in diagnostic['raw_task_gradients'].values())
                if rank==0:
                    results[-1]['diagnostic_graph_protocol']=diagnostic['graph_protocol']
                    print(json.dumps(dict(diagnostic_passed=True,affine=affine,phase=phase,execution=execution)),flush=True)
                fitted=fit(bank,core,settings,device);metrics=evaluate(observer,core,bank,fitted,settings,device)
                if rank==0:
                    with patch('emotion_ssm.train.staged_v33.evaluation.dist.is_initialized',lambda:False):
                        single=evaluate(observer,core,bank,fitted,{**settings,'prediction_batch':3},device)
                    for name in metrics['methods']:
                        a=metrics['methods'][name];b=single['methods'][name]
                        assert a['elements']==b['elements']
                        assert abs(a['macro_mse']-b['macro_mse'])<max(1e-8,abs(b['macro_mse'])*1e-5)
                        assert metrics['semantic'][name]['query_weighted']['emotion_count']==single['semantic'][name]['query_weighted']['emotion_count']
                dist.barrier()
    if rank==0:(root/'result.json').write_text(json.dumps(dict(status='passed',checks=results),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':main()
