import copy
import torch
from test_staged_v34 import case
from scripts.diagnose_staged_v34_origins import trajectories


def test_role_specific_relation_and_zero_second_trajectory(tmp_path):
    torch.set_num_threads(1)
    _,o,t,c,col,cache,bank,s=case(tmp_path)
    bank=copy.deepcopy(bank)
    bank['endpoints']=[(r,{**q,'nominal_horizon':32,'seconds':32.25}) for r,q in bank['endpoints'] if q['nominal_horizon']==2]
    values=trajectories(o,c,bank,torch.zeros(4),'cpu',batch=1)
    assert len(values)==len(bank['endpoints'])*7
    assert {v['horizon'] for v in values}=={0,1,2,4,8,16,32}
    assert all(isinstance(v['relation_norm'],float) for v in values)
    assert all(v['actual_seconds']==32.25 for v in values if v['horizon']==32)
    assert all(abs(sum(v['probability'])-1)<1e-6 for v in values)
