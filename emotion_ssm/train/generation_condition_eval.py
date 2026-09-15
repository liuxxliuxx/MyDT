"""Causal condition interventions with one shared interaction feature per block.

Donors/means are fitted on train only. Reconstructions under a wrong condition
measure sensitivity, not whether that alternative motion is semantically correct.
"""
from __future__ import annotations
import bisect
import hashlib
import json
from pathlib import Path
import torch
from emotion_ssm.models.streaming_v3 import map_tensors
from emotion_ssm.models.visual_affect_teacher import observer_digest
from emotion_ssm.train.generation_sampling import source_id
from emotion_ssm.utils.reconstruction import ReconstructionTotals
from emotion_ssm.utils.generation_losses import matching_boundary,boundary_frame

PROTOCOL='causal-fixed-feature-conditions-v1'
MODES=('true','train_mean','matched_train','film_off')


def coordinate_identity(model):
    route=getattr(model,'condition_router',None)
    if route is not None and route.mode!='full_state':
        raise ValueError('This state-condition intervention protocol requires full-state routing')
    if model.variant not in ('self','dyadic'):
        raise ValueError('Condition interventions require persistent self/dyadic state formation')
    return dict(observer=observer_digest(model.observer),state=observer_digest(model.state_model),
                variant=model.variant,context_dim=model.state_model.context_dim)


def indices_for(dataset,names):
    lookup={name:i for i,name in enumerate(dataset.names)}
    if len(set(names))!=len(names) or set(names)-set(lookup):
        raise ValueError('Fixed evaluation manifest contains duplicates or unavailable records')
    return [lookup[name] for name in names]


@torch.no_grad()
def fit_training_conditions(model,dataset,names,device='cpu',anchor_seconds=4.):
    if dataset.split!='train':raise ValueError('Fit condition means and donors on train only')
    model.eval();identity=coordinate_identity(model);rows=[];total=None;frames=0
    sources=getattr(dataset,'source_ids',[source_id(n) for n in dataset.names])
    for index in indices_for(dataset,names):
        state=None;contexts=[];times=[];anchors=[];anchor_duration=0.
        for packet,target,valid in dataset.packets(index):
            packet=map_tensors(packet,lambda x:x.to(device))
            _,state,d=model(packet,state,observe_only=True)
            context=d['generator_context'].detach().cpu().reshape(-1)
            age=packet['time']-packet.get('start_time',0.)
            contexts.append(context);times.append(age)
            count=int(valid.sum())
            total=context.double()*count if total is None else total+context.double()*count
            frames+=count
            if age<=anchor_seconds+1e-6:
                dt=target.shape[1]/25
                anchors.append((d['target_aff'].cpu().reshape(-1)*dt,d['partner_aff'].cpu().reshape(-1)*dt))
                anchor_duration+=dt
        if times and times[-1]>=anchor_seconds and anchors:
            anchor=torch.cat([sum(a[role] for a in anchors)/anchor_duration for role in (0,1)])
            rows.append(dict(name=dataset.names[index],source=sources[index],times=times,
                             contexts=torch.stack(contexts),anchor=anchor))
    if not frames or not rows:raise ValueError('Insufficient valid train context/donor coverage')
    return dict(protocol=PROTOCOL,split='train',coordinate=identity,manifest_digest=dataset.manifest_digest,
                names=list(names),anchor_seconds=anchor_seconds,mean=(total/frames).float(),valid_frames=frames,donors=rows)


@torch.no_grad()
def evaluate_conditions(model,dataset,names,bank,device='cpu',score_start_seconds=4.,history_ablations=False):
    if bank['protocol']!=PROTOCOL or bank['split']!='train' or bank['coordinate']!=coordinate_identity(model):
        raise ValueError('Condition bank does not match this frozen coordinate/state producer')
    if score_start_seconds<bank['anchor_seconds']:
        raise ValueError('Score only after the causal matching anchor is fully visible')
    from emotion_ssm.train.history_condition_eval import HistoryConditionReplay
    modes=MODES+HistoryConditionReplay.modes if history_ablations else MODES
    model.eval();totals={m:ReconstructionTotals() for m in modes};records=[];paired_true=ReconstructionTotals()
    raw_modulation=[0.,0.,0]
    sources=getattr(dataset,'source_ids',[source_id(n) for n in dataset.names])
    if dataset.split!='train' and set(sources)&{d['source'] for d in bank['donors']}:
        raise ValueError('Evaluation and donor train sources overlap')
    teacher=getattr(model,'visual_teacher',None)
    visual_ok=False
    if teacher is not None:
        try:teacher.require_validated(['affect']);visual_ok=True
        except ValueError:pass
    mean=bank['mean'].to(device);coverage=possible=0
    for index in indices_for(dataset,names):
        state=None;histories={m:[] for m in modes};edges={m:None for m in modes}
        replay=HistoryConditionReplay(model.state_model,model.variant) if history_ablations else None
        local={m:ReconstructionTotals() for m in modes};sens={m:dict(condition_sse=0.,condition_elements=0,
            output_sse=0.,output_elements=0,gamma_sse=0.,beta_sse=0.,modulation_elements=0,
            feature_sse=0.,feature_elements=0,visual_cosine_sum=0.,visual_windows=0) for m in modes}
        anchors=[];anchor_duration=0.;donor=None;matching_closed=False;previous_gamma=None;temporal_sse=0.;temporal_n=0
        for raw,target,valid in dataset.packets(index):
            packet=map_tensors(raw,lambda x:x.to(device));target=target.to(device);valid=valid.to(device).bool()
            _,state,d=model(packet,state,observe_only=True,return_generator_inputs=True)
            a,b,v,true_context=d['generator_inputs'];n=target.shape[1]
            age=packet['time']-packet.get('start_time',0.)
            if age<=bank['anchor_seconds']+1e-6:
                dt=n/25;anchors.append(torch.cat([d['target_aff'].reshape(-1),d['partner_aff'].reshape(-1)])*dt)
                anchor_duration+=dt
            if not matching_closed and age>=bank['anchor_seconds']:
                matching_closed=True
                eligible=[row for row in bank['donors'] if row['source']!=sources[index]]
                if eligible and anchors:
                    anchor=sum(anchors)/anchor_duration
                    donor=min(eligible,key=lambda row:(float((row['anchor'].to(device)-anchor).square().mean()),row['name']))
            eligible_donor=donor is not None and age<=donor['times'][-1]+1e-6
            current=d['generator_context'].reshape(1,1,-1).expand(-1,n,-1)
            alternatives=dict(true=current,train_mean=mean.reshape(1,1,-1).expand_as(current),film_off=current)
            if eligible_donor:
                j=bisect.bisect_right(donor['times'],age+1e-6)-1
                alternatives['matched_train']=donor['contexts'][max(0,j)].to(device).reshape(1,1,-1).expand_as(current)
            else:alternatives['matched_train']=current
            if replay is not None:
                alternatives.update({mode:context[:,None].expand(-1,n,-1) for mode,context in
                    replay.advance(d['observations'],packet['time'],n/25).items()})
            # Encode once: only the conditioning/synthesis path changes.
            features=model.generator.encode_interaction(a,b,v)
            outputs={};modulations={};condition_sequences={}
            for mode,new in alternatives.items():
                seq=torch.cat(histories[mode]+[new],1);condition_sequences[mode]=seq
                outputs[mode]=model.generator.decode_interaction(features,seq,mode!='film_off')[:,-n:]
                gamma,beta=model.generator.film.modulation(seq)
                if mode=='film_off':gamma=torch.zeros_like(gamma);beta=torch.zeros_like(beta)
                modulations[mode]=(gamma,beta)
                histories[mode]=[seq[:,-model.history_frames:]] if model.history_frames else []
            if age<=score_start_seconds+1e-6:
                # Matching may start at 4s; these unscored causal warmup blocks
                # do not contribute the first scored boundary either.
                continue
            possible+=1;coverage+=int(eligible_donor)
            true_visual=teacher(outputs['true'],valid,domain_id=packet.get('source_domain',2),now=packet['time']) if visual_ok else None
            gamma,beta=modulations['true']
            raw_modulation[0]+=float(gamma.double().square().sum());raw_modulation[1]+=float(beta.double().square().sum())
            raw_modulation[2]+=gamma.numel()
            current_gamma=gamma[:,-n:].mean(1)
            if previous_gamma is not None:
                temporal_sse+=float((current_gamma-previous_gamma).double().square().sum());temporal_n+=current_gamma.numel()
            previous_gamma=current_gamma
            for mode,pred in outputs.items():
                if mode=='matched_train' and not eligible_donor:
                    edges[mode]=None;continue
                edge=matching_boundary(packet,n,edges[mode]);prev=None if edge is None else (edge.prediction,edge.target,edge.valid)
                if mode=='true' and eligible_donor:paired_true.update(pred,target,valid,prev)
                local[mode].update(pred,target,valid,prev);edges[mode]=boundary_frame(packet,pred,target,valid)
                stat=sens[mode];c=condition_sequences[mode]-condition_sequences['true'];g,be=modulations[mode]
                perturbed=features*(1+g)+be;base=features*(1+gamma)+beta
                for key,error in [('condition',c),('output',(pred-outputs['true'])[valid]),
                                  ('feature',perturbed-base)]:
                    stat[key+'_sse']+=float(error.double().square().sum());stat[key+'_elements']+=error.numel()
                stat['gamma_sse']+=float((g-gamma).double().square().sum());stat['beta_sse']+=float((be-beta).double().square().sum())
                stat['modulation_elements']+=g.numel()
                if visual_ok and valid.any():
                    visual=teacher(pred,valid,domain_id=packet.get('source_domain',2),now=packet['time'])
                    stat['visual_cosine_sum']+=float((1-torch.nn.functional.cosine_similarity(visual['affect'],true_visual['affect'])).sum())
                    stat['visual_windows']+=len(pred)
        for mode in modes:
            totals[mode].merge(local[mode]);records.append(dict(dialogue=dataset.names[index],source=sources[index],mode=mode,
                donor=None if donor is None else donor['name'],metrics=local[mode].metrics(),sensitivity=sens[mode],
                temporal_gamma_mse=temporal_sse/max(1,temporal_n),sse=dict(local[mode].square_error),elements=dict(local[mode].elements)))
    sensitivity={}
    for mode in modes:
        aggregate={k:sum(row['sensitivity'][k] for row in records if row['mode']==mode) for k in next(r['sensitivity'] for r in records if r['mode']==mode)}
        sensitivity[mode]={key+'_mse':aggregate[key+'_sse']/max(1,aggregate[key+'_elements']) for key in ('condition','output','feature')}
        sensitivity[mode].update(gamma_mse=aggregate['gamma_sse']/max(1,aggregate['modulation_elements']),
            beta_mse=aggregate['beta_sse']/max(1,aggregate['modulation_elements']),
            visual_cosine_distance=aggregate['visual_cosine_sum']/aggregate['visual_windows'] if aggregate['visual_windows'] else None)
    return dict(protocol=PROTOCOL,split=dataset.split,manifest_digest=dataset.manifest_digest,names=list(names),
        coordinate=bank['coordinate'],score_start_seconds=score_start_seconds,matching_coverage=coverage/max(1,possible),
        matched_blocks=coverage,candidate_blocks=possible,metrics={m:totals[m].metrics() for m in modes},
        history_protocol='Shared causal observer inputs (may contain 16s acoustic history); truncate persistent state only; uncoupled replays all history with coupling disabled.',
        sensitivity=sensitivity,records=records,independent_semantics=None,
        paired_true_on_matched_coverage=paired_true.metrics(),
        true_modulation=dict(gamma_rms=(raw_modulation[0]/max(1,raw_modulation[2]))**.5,
                             beta_rms=(raw_modulation[1]/max(1,raw_modulation[2]))**.5),
        interpretation='Wrong-condition motion MSE measures sensitivity; it is not semantic correctness.')


def main():
    import argparse
    from emotion_ssm.utils.checkpoint_v3 import load_avatar
    from emotion_ssm.train.generation_v3 import GenerationTokenDataset
    parser=argparse.ArgumentParser();parser.add_argument('--checkpoint',required=True);parser.add_argument('--manifest',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu');parser.add_argument('--bank')
    parser.add_argument('--history-ablations',action='store_true')
    args=parser.parse_args();model,cfg,payload=load_avatar(args.checkpoint,args.device)
    manifest=json.loads(Path(args.manifest).read_text(encoding='utf-8'));split=manifest['split']
    data=GenerationTokenDataset(cfg['data']['dualtalk_tokens'],cfg['data']['dualtalk_raw'],split)
    timeline=cfg['data'].get('continuous_timelines',{}).get(split)
    if timeline:
        from emotion_ssm.data.continuous_dialogues import ContinuousDialogueDataset
        data=ContinuousDialogueDataset(data,timeline)
    if manifest['dataset_digest']!=data.manifest_digest:raise ValueError('Evaluation population changed')
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    if args.bank:
        bank=torch.load(args.bank,map_location='cpu',weights_only=False)
        report=evaluate_conditions(model,data,manifest['names'],bank,args.device,manifest.get('score_start_seconds',4.),args.history_ablations)
        report['checkpoint_step']=payload['global_step']
        output.write_text(json.dumps(report,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    else:
        bank=fit_training_conditions(model,data,manifest['names'],args.device)
        torch.save(bank,output)
        config=dict(mode='mean',train_mean=bank['mean'].tolist(),mean_provenance=dict(split='train',
            manifest_digest=bank['manifest_digest'],coordinate=bank['coordinate'],names=bank['names']))
        output.with_suffix('.mean.json').write_text(json.dumps(config,indent=2),encoding='utf-8')


if __name__=='__main__':main()
