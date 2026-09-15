"""Prepare a new v3.4 experiment using long-history and real-time missing views."""
import argparse
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emotion_ssm.config_v3 import read_config,write_config


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--base-config',required=True)
    p.add_argument('--initialize',required=True);p.add_argument('--output-config',required=True);p.add_argument('--run-directory',required=True)
    a=p.parse_args();cfg=read_config(a.base_config)
    cfg['paths'].update(resume='',dynamics_checkpoint=a.initialize,output=a.run_directory)
    cfg.setdefault('staged',{}).update(flow_replay_origins=True,full_origin_replay=True,origin_refresh_steps=250,origin_probe_steps=100,
        history_age_sampling=dict(age_edges=[0,8,16,32,64],natural_probability=.5),
        train_protocol_cycle=['clean','clean','clean','train_context'],block_gap_period=64.,block_gap_seconds=16.)
    cfg['experiment']=dict(protocol='long-history-physical-context-v1',initialization=a.initialize,
        description='Clean primary view, separate temporal sensor outages, exact current origin replay, fixed correction strengths')
    write_config(a.output_config,cfg)
    print('CUDA_VISIBLE_DEVICES=2,3 python -m torch.distributed.run --standalone --nproc_per_node=2 -m emotion_ssm.train.staged_v34.trainer --config '+str(a.output_config))


if __name__=='__main__':main()
