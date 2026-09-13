"""Train-only unique-endpoint class weights and explicit neutral confusion metrics."""
import torch
from emotion_ssm.train.staged_v33.missing import digest

NEUTRAL=4


def class_audit(bank,settings):
    counts=torch.zeros(int(bank['domain'].max())+1,len(bank['horizons']),7,dtype=torch.float64)
    seen=set()
    for row,q in bank['endpoints']:
        label=int(q['label'].get('emotion',-1))
        if not 0<=label<7:continue
        d=int(bank['domain'][row]);h=bank['horizons'].index(q['nominal_horizon'])
        key=(bank['keys'][row][0],q['role'],q['endpoint'],h)
        if key in seen:raise ValueError('Repeated class label at the same real endpoint/horizon')
        seen.add(key);counts[d,h,label]+=1
    power=float(settings.get('class_weight_power',.5));cap=float(settings.get('class_weight_cap',3.))
    if not 0<=power<=1 or cap<1:raise ValueError('Invalid bounded class weighting')
    present=counts>0
    raw=counts.clamp_min(1).pow(-power)*present
    mean=(raw*counts).sum(-1,keepdim=True)/counts.sum(-1,keepdim=True).clamp_min(1)
    weights=(raw/mean.clamp_min(1e-12)).clamp(max=cap)*present
    # No weighted random sampler is added: these are the only class weights.
    return dict(scope='current_training_bank_unique_endpoint_per_domain_horizon',fit_split='train',
        counts=counts.tolist(),weights=weights.tolist(),power=power,cap=cap,
        endpoints=len({(k[0],k[1],k[2]) for k in seen}),horizon_queries=len(seen),
        labels_hash=digest(dict(keys=sorted(seen),counts=counts.tolist())),source_dialogues=[e['identity'] for e in bank['encoded_rows']])


def confusion_metrics(cm):
    cm=torch.as_tensor(cm,dtype=torch.float64)
    truth=cm.sum(-1);pred=cm.sum(-2);total=float(cm.sum());neutral=float(truth[NEUTRAL]);other=total-neutral
    ratio=lambda a,b:float(a/b) if float(b)>0 else None
    return dict(count=total,truth_counts=truth.tolist(),prediction_counts=pred.tolist(),
        per_class_recall=[ratio(cm[i,i],truth[i]) for i in range(7)],
        nonneutral_to_neutral=ratio(pred[NEUTRAL]-cm[NEUTRAL,NEUTRAL],other),
        neutral_to_nonneutral=ratio(neutral-cm[NEUTRAL,NEUTRAL],neutral),
        neutral_precision=ratio(cm[NEUTRAL,NEUTRAL],pred[NEUTRAL]),
        neutral_recall=ratio(cm[NEUTRAL,NEUTRAL],neutral),neutral_prediction_share=ratio(pred[NEUTRAL],total))
