"""Audit actual history/horizon support or prepare a non-executable timeline template."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    from emotion_ssm.config_v3 import read_config
    from emotion_ssm.utils.history_coverage import endpoint_coverage
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--output',required=True);p.add_argument('--split',choices=['train','val'],default='train')
    p.add_argument('--avatar-timeline-template',action='store_true')
    a=p.parse_args();cfg=read_config(a.config)
    if a.avatar_timeline_template:
        from emotion_ssm.train.generation_v3 import GenerationTokenDataset
        from emotion_ssm.data.continuous_dialogues import PROTOCOL
        data=GenerationTokenDataset(cfg['data']['dualtalk_tokens'],cfg['data']['dualtalk_raw'],a.split)
        result=dict(protocol=PROTOCOL,split=a.split,token_manifest_digest=data.manifest_digest,conversations=[],
            status='awaiting_original_timestamps_and_verified_ordered_roles',unmapped_records=data.names,
            example=dict(session_id='original-conversation-id',source_id='original-video-id',roles=['avatar-person','user-person'],
                segments=[dict(name='record-from-unmapped_records',start=0.,end=0.,roles=['avatar-person','user-person'],
                               verified=False,evidence='Original metadata/video timestamp and identity evidence required')]))
    else:
        from emotion_ssm.train.dynamics_v3 import DialogueCollection
        data=DialogueCollection(cfg['data']['token_roots'],a.split)
        result=endpoint_coverage(((data.identity(i),data[i]['packets']) for i in range(len(data))),cfg['train']['forecast_seconds'])
        result['split']=a.split
    path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    print(str(path))


if __name__=='__main__':main()
