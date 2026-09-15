"""Train FLAME adapters on independent human labels and validate on held sources."""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import torch
from torch.nn import functional as F

from emotion_ssm.models.visual_affect_teacher import FrozenFlameAffect, FLAME_SCHEMA, observer_digest,affect_coordinate_digest
from emotion_ssm.metrics import confusion_matrix, classification_metrics


def load_rows(path, split):
    path=Path(path);data=json.loads(path.read_text(encoding='utf-8'))
    if data.get('protocol')!='human-flame-calibration-v1' or data.get('split')!=split or data.get('motion_schema')!=FLAME_SCHEMA:
        raise ValueError('Calibration requires an explicit human-label split and motion schema')
    rows=[]
    from emotion_ssm.data.common import EMOTION_NAMES
    if data.get('emotion_classes')!=list(EMOTION_NAMES):
        raise ValueError('Declare the exact ordered seven-class label ontology')
    for item in data.get('samples',[]):
        if item.get('annotation_origin')!='human_manual' or not item.get('source_id') or not item.get('annotator_ids'):
            raise ValueError('Model predictions and software tests cannot be calibration gold labels')
        motion_path=Path(item['motion_path'])
        if not motion_path.is_absolute(): motion_path=path.parent/motion_path
        payload=torch.load(motion_path,map_location='cpu',weights_only=False)
        motion=payload['motion'];mask=payload.get('frame_mask',torch.ones(motion.shape[:-1],dtype=torch.bool))
        if motion.ndim==2: motion=motion[None];mask=mask[None]
        if motion.shape[0]!=1 or motion.shape[-1]!=56 or mask.shape!=motion.shape[:2] or not mask.any():
            raise ValueError('Calibration rows need one valid FLAME window')
        label=dict(item.get('labels',{}))
        emotion=int(label.get('emotion',-1))
        if emotion not in range(-1,7): raise ValueError('Emotion labels must use the declared seven-class ontology')
        vad=torch.tensor(label.get('vad',[0.,0.,0.]),dtype=torch.float32)
        vm=torch.tensor(label.get('vad_mask',[False]*3),dtype=torch.bool)
        if vad.shape!=(3,) or vm.shape!=(3,) or not torch.isfinite(vad[vm]).all() or (vad[vm].abs()>1).any():
            raise ValueError('VAD coordinates require explicit [-1,1] human-label normalization')
        if label.get('intensity_mask') and label.get('intensity_scale')!='teacher_calibrated':
            raise ValueError('Ordinal human intensity cannot be silently treated as teacher-scale regression')
        if label.get('intensity_mask') and (not torch.isfinite(torch.tensor(label['intensity'])) or label['intensity']<0):
            raise ValueError('Invalid calibrated intensity label')
        rows.append(dict(item,motion=motion,mask=mask,emotion=emotion,vad=vad,vad_mask=vm,
                         intensity=float(label.get('intensity',0.)),intensity_mask=bool(label.get('intensity_mask',False))))
    return rows,hashlib.sha256(path.read_bytes()).hexdigest()


def supervised(output,row):
    total=output['affect'].sum()*0.
    if row['emotion']>=0:
        total=total+F.cross_entropy(output['emotion_logits'],torch.tensor([row['emotion']],device=output['affect'].device))
    mask=row['vad_mask'].to(output['affect'].device)
    if mask.any(): total=total+(output['vad'][0,mask]-row['vad'].to(mask.device)[mask]).square().mean()
    if row['intensity_mask']: total=total+(output['intensity'][0]-row['intensity']).square()
    return total


@torch.no_grad()
def score(teacher,rows,device):
    outputs=[teacher(r['motion'].to(device),r['mask'].to(device),domain_id=r.get('domain_id',2)) for r in rows]
    labelled=[i for i,r in enumerate(rows) if r['emotion']>=0]
    result=dict(rows=len(rows),gold_rows=len(labelled),loss=None,emotion=None,vad_mse=[None]*3,intensity_mse=None)
    if rows:
        result['loss']=sum(float(supervised(o,r)) for o,r in zip(outputs,rows))/len(rows)
    if labelled:
        y=torch.tensor([rows[i]['emotion'] for i in labelled]);p=torch.tensor([int(outputs[i]['emotion_logits'].argmax(-1)) for i in labelled])
        cm=confusion_matrix(y,p);result['emotion']=dict(classification_metrics(cm),confusion=cm.tolist(),classes_present=int((cm.sum(1)>0).sum()))
    for d in range(3):
        errors=[float((o['vad'][0,d]-r['vad'][d]).square()) for o,r in zip(outputs,rows) if r['vad_mask'][d]]
        if errors: result['vad_mse'][d]=sum(errors)/len(errors)
    errors=[float((o['intensity'][0]-r['intensity']).square()) for o,r in zip(outputs,rows) if r['intensity_mask']]
    if errors: result['intensity_mse']=sum(errors)/len(errors)
    return result,outputs


def calibrate(observer,train,val,*,steps=500,lr=1e-4,device='cpu',minimum_f1_gain=.01):
    if not train or not val:
        return None,dict(status='awaiting_independent_human_labels',train_rows=len(train),validation_rows=len(val))
    if {r['source_id'] for r in train}&{r['source_id'] for r in val}:
        raise ValueError('Calibration train/val source identities overlap')
    if any(r.get('annotation_origin')!='human_manual' for r in train+val):
        raise ValueError('Calibration requires human labels, not pseudo labels')
    domains=sorted({int(r.get('domain_id',2)) for r in train})
    if any(int(r.get('domain_id',2)) not in domains for r in val):
        raise ValueError('Validation has a visual source not trained by this calibration')
    teacher=FrozenFlameAffect(observer,dict(domains=domains)).to(device)
    before,_=score(teacher,val,device)
    # Shared affect coordinates and all label heads remain frozen.
    parameters=[]
    for domain in domains:
        adapter=teacher.observer.adapters['flame'][domain]
        adapter.requires_grad_(True);parameters.extend(adapter.parameters())
    optimizer=torch.optim.AdamW(parameters,lr=lr,weight_decay=.001)
    best=float('inf');best_weights=None
    for step in range(steps):
        row=train[step%len(train)]
        optimizer.zero_grad(set_to_none=True)
        out=teacher(row['motion'].to(device),row['mask'].to(device),domain_id=row.get('domain_id',2))
        loss=supervised(out,row)
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite visual calibration loss')
        loss.backward();torch.nn.utils.clip_grad_norm_(parameters,1.);optimizer.step()
        if (step+1)%max(1,min(50,steps))==0 or step+1==steps:
            current,_=score(teacher,val,device)
            if current['loss']<best:
                best=current['loss'];best_weights=copy.deepcopy(teacher.state_dict())
    if best_weights is not None: teacher.load_state_dict(best_weights)
    teacher.requires_grad_(False);after,outputs=score(teacher,val,device)
    row=val[0];motion=row['motion'].to(device).clone().requires_grad_(True)
    out=teacher(motion,row['mask'].to(device),domain_id=row.get('domain_id',2))
    # Unit-vector norm is constant; use a fixed coordinate probe instead.
    grad=torch.autograd.grad(out['affect'][0,0],motion)[0]
    gradient_ok=bool(torch.isfinite(grad).all() and grad.abs().sum()>1e-10)
    with torch.no_grad():
        mean_motion=motion.detach().mean(1,keepdim=True).expand_as(motion)
        response=float((teacher(mean_motion,row['mask'].to(device),domain_id=row.get('domain_id',2))['affect']-out['affect']).square().mean())
    labelled=[i for i,r in enumerate(val) if r['emotion']>=0]
    pair_signals=[]
    for position,i in enumerate(labelled):
        for j in labelled[position+1:]:
            a,b=val[i]['emotion'],val[j]['emotion']
            if a==b:continue
            logits_i,logits_j=outputs[i]['emotion_logits'][0],outputs[j]['emotion_logits'][0]
            pair_signals.append(float((logits_i[a]-logits_j[a])+(logits_j[b]-logits_i[b]))>0)
    pair_agreement=sum(pair_signals)/len(pair_signals) if pair_signals else None
    validated=[];control=None
    train_classes=[r['emotion'] for r in train if r['emotion']>=0]
    if train_classes and after['emotion'] is not None:
        majority=max(set(train_classes),key=train_classes.count)
        target=torch.tensor([r['emotion'] for r in val if r['emotion']>=0])
        control=classification_metrics(confusion_matrix(target,torch.full_like(target,majority)))
        if (after['emotion']['classes_present']>=2 and after['emotion']['macro_f1']>=control['macro_f1']+minimum_f1_gain
                and response>1e-10 and gradient_ok and pair_agreement is not None and pair_agreement>.5):
            validated.extend(['affect','emotion'])
    vad_controls=[]
    for d in range(3):
        tr=[float(r['vad'][d]) for r in train if r['vad_mask'][d]]
        va=[float(r['vad'][d]) for r in val if r['vad_mask'][d]]
        control_d=sum((x-sum(tr)/len(tr))**2 for x in va)/len(va) if tr and len(va)>=2 else None
        vad_controls.append(control_d)
    if all(c is not None and c>0 and after['vad_mse'][d] is not None and after['vad_mse'][d]<c for d,c in enumerate(vad_controls)) and gradient_ok:
        validated.extend(['affect','vad'])
    if gradient_ok:
        for d,c in enumerate(vad_controls):
            if c is not None and c>0 and after['vad_mse'][d] is not None and after['vad_mse'][d]<c:
                validated.extend(['affect','vad:'+str(d)])
    tr=[r['intensity'] for r in train if r['intensity_mask']];va=[r['intensity'] for r in val if r['intensity_mask']]
    ic=sum((x-sum(tr)/len(tr))**2 for x in va)/len(va) if tr and len(va)>=2 else None
    if ic is not None and ic>0 and after['intensity_mse']<ic and gradient_ok: validated.append('intensity')
    report=dict(status='passed' if validated else 'failed_validation',validated_heads=sorted(set(validated)),
                independent_human_rows=len(val),calibration_split='train',validation_split='val',source_disjoint=True,
                motion_schema=FLAME_SCHEMA,input_gradient_finite_nonzero=gradient_ok,mean_motion_response_mse=response,
                teacher_sha256=observer_digest(teacher.observer),affect_coordinate_sha256=affect_coordinate_digest(teacher.observer),before=before,after=after,
                different_emotion_pair_agreement=pair_agreement,different_emotion_pairs=len(pair_signals),
                majority_control=control,train_mean_vad_controls=vad_controls,train_mean_intensity_control=ic,
                minimum_f1_gain=minimum_f1_gain,steps=steps,
                note='Validation is for teacher gating; use separate untouched sources for final Avatar semantics.')
    teacher.validation=report
    return teacher,report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint',required=True);p.add_argument('--train-manifest',required=True)
    p.add_argument('--val-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--steps',type=int,default=500);p.add_argument('--device',default='cpu');p.add_argument('--lr',type=float,default=1e-4)
    a=p.parse_args()
    from emotion_ssm.utils.checkpoint_v3 import load_avatar
    model,_,_=load_avatar(a.checkpoint,'cpu')
    train,td=load_rows(a.train_manifest,'train');val,vd=load_rows(a.val_manifest,'val')
    teacher,report=calibrate(model.observer,train,val,steps=a.steps,lr=a.lr,device=a.device)
    report.update(train_manifest_sha256=td,val_manifest_sha256=vd)
    output=Path(a.output);output.mkdir(parents=True,exist_ok=True)
    (output/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if teacher is not None:
        teacher.validation=report
        torch.save(dict(kind='validated_flame_teacher_v1',construction=teacher.construction(),
                        state_dict=teacher.state_dict()),output/('teacher.pt' if report['status']=='passed' else 'candidate_unvalidated.pt'))
    print(json.dumps(report))


if __name__=='__main__': main()
