"""Export train/val FLAME windows with blank human-label fields; never pseudo-label."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from emotion_ssm.data.common import EMOTION_NAMES
from emotion_ssm.models.visual_affect_teacher import FLAME_SCHEMA


def template(split):
    return dict(protocol='human-flame-calibration-v1',split=split,motion_schema=FLAME_SCHEMA,
                emotion_classes=list(EMOTION_NAMES),status='awaiting_independent_human_labels',samples=[])


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',required=True)
    p.add_argument('--config');p.add_argument('--selection');p.add_argument('--templates-only',action='store_true')
    a=p.parse_args();output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    manifests={split:template(split) for split in ('train','val')}
    if not a.templates_only:
        if not a.config or not a.selection:raise ValueError('Provide a fixed train/val selection before exporting motion')
        from emotion_ssm.config_v3 import read_config
        from emotion_ssm.train.generation_v3 import GenerationTokenDataset
        cfg=read_config(a.config);selection=json.loads(Path(a.selection).read_text(encoding='utf-8'))
        if set(selection)!=set(manifests):raise ValueError('Calibration selection must contain train and val only; never test/OOD')
        used_sources={}
        from emotion_ssm.train.generation_sampling import source_id
        for split,spec in selection.items():
            data=GenerationTokenDataset(cfg['data']['dualtalk_tokens'],cfg['data']['dualtalk_raw'],split)
            if spec['dataset_digest']!=data.manifest_digest:raise ValueError('Calibration source manifest changed')
            manifests[split]['dataset_digest']=data.manifest_digest
            by_name={name:i for i,name in enumerate(data.names)};used_sources[split]=set();seen=set()
            for window in spec['windows']:
                name=window['name'];end=float(window['end']);start=float(window.get('start',end-1.))
                if (name,start,end) in seen:raise ValueError('Repeated calibration window')
                seen.add((name,start,end))
                if end<=start or end-start>1.000001:raise ValueError('First calibration protocol uses at most one causal second')
                motion=[];masks=[]
                for packet,target,valid in data.packets(by_name[name]):
                    times=torch.arange(target.shape[1])/25+packet['time']-target.shape[1]/25
                    keep=(times>=start-1e-6)&(times<end-1e-6)
                    if keep.any():motion.append(target[:,keep]);masks.append(valid[:,keep])
                if not motion:raise ValueError('No frames for the selected calibration interval')
                sid=hashlib.sha256(f'{split}:{name}:{start}:{end}'.encode()).hexdigest()[:16]
                path=output/(sid+'.pt');torch.save(dict(motion=torch.cat(motion,1),frame_mask=torch.cat(masks,1)),path)
                source=source_id(name);used_sources[split].add(source)
                manifests[split]['samples'].append(dict(sample_id=sid,motion_path=path.name,source_id=source,record=name,
                    start=start,end=end,domain_id=2,annotation_origin='unannotated',annotator_ids=[],
                    labels=dict(emotion=-1,vad=[0.,0.,0.],vad_mask=[False,False,False],intensity=0.,intensity_mask=False)))
        if used_sources['train']&used_sources['val']:raise ValueError('Calibration source identities overlap')
    for split,manifest in manifests.items():
        (output/(split+'.json')).write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding='utf-8')
    status=dict(status='awaiting_independent_human_labels',human_rows=0,
        exported_windows={split:len(m['samples']) for split,m in manifests.items()},final_test_used=False,visual_losses_enabled=False)
    (output/'status.json').write_text(json.dumps(status,indent=2),encoding='utf-8');print(json.dumps(status))


if __name__=='__main__':main()
