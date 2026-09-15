"""Promote a validation candidate only after independent semantic assessment."""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def file_digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()


def select(checkpoint,independent,reference,output):
    from emotion_ssm.utils.checkpoint_v3 import read_checkpoint
    from emotion_ssm.utils.generation_selection import conditioned_acceptance
    payload=read_checkpoint(checkpoint)
    if not payload.get('experiment',{}).get('deployable',True):raise ValueError('Oracle diagnostics cannot become deployment checkpoints')
    if not payload['config'].get('generation_selection',{}).get('conditioned_policy'):
        raise ValueError('No semantic acceptance policy was declared before training')
    identity=dict(checkpoint_sha256=file_digest(checkpoint),provenance=payload['provenance'])
    if reference.get('identity')!=independent.get('reference_identity') or reference.get('split')!='val':
        raise ValueError('Independent report and reconstruction control must reference the same validation control')
    if reference.get('population_sha256')!=independent.get('population_sha256') or not independent.get('population_sha256'):
        raise ValueError('Paired validation population is missing or inconsistent')
    policy=payload['config']['generation_selection']['conditioned_policy']
    if policy.get('population_sha256')!=independent['population_sha256']:
        raise ValueError('Semantic selection population differs from the predeclared policy')
    result=conditioned_acceptance(payload['metrics'],reference['metrics'],policy,independent,identity)
    result.update(identity=identity,population_sha256=independent['population_sha256'])
    out=Path(output);out.mkdir(parents=True,exist_ok=True)
    (out/'conditioned_selection.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    if result['accepted']:
        target=out/'best_conditioned.pt'
        # A later accepted candidate must also improve the same independent metric.
        previous=out/'best_conditioned_semantics.json';metric=policy['semantic_metric']
        better=not previous.exists() or independent['improvements'][metric]>json.loads(previous.read_text())['improvements'][metric]
        if better:
            temporary=target.with_suffix('.tmp');shutil.copyfile(checkpoint,temporary);temporary.replace(target)
            previous.write_text(json.dumps(independent,indent=2),encoding='utf-8')
        result['promoted']=better
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',required=True)
    p.add_argument('--independent-report',required=True);p.add_argument('--reference',required=True);p.add_argument('--output',required=True)
    a=p.parse_args();print(json.dumps(select(a.checkpoint,json.loads(Path(a.independent_report).read_text()),
        json.loads(Path(a.reference).read_text()),a.output)))


if __name__=='__main__':main()
