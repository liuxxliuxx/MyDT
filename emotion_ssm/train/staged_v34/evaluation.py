"""Per-domain/per-horizon semantic and neutral gates on exact online replays."""
import math
import torch
from emotion_ssm.train.staged_v33.evaluation import evaluate as base_evaluate,semantic_summary
from .semantics import confusion_metrics


def enrich(result):
    semantic=torch.tensor(result['semantic_sums'],dtype=torch.float64)
    cms=torch.tensor(result['semantic_confusions'],dtype=torch.float64)
    result['semantic_cells']={}
    for m,name in enumerate(result['methods_order']):
        cells={}
        for d in range(semantic.shape[1]):
            for h,seconds in enumerate(result['horizons']):
                values=semantic_summary(semantic[m,d:d+1,h:h+1],cms[m,d:d+1,h:h+1])['query_weighted']
                cells[f'{d}/{seconds:g}']={**values,**confusion_metrics(cms[m,d,h,0])}
        result['semantic_cells'][name]=cells
    return result


def evaluate(*args,**kwargs):return enrich(base_evaluate(*args,**kwargs))


def deployment_gate(metrics,initial,settings):
    checks={};details={};delta=settings.get('long_semantic_tolerance',.01)
    relative=settings.get('acceptance_tolerance',.02)
    for view in ('live','update_gap','sensor_gap'):
        current=enrich(metrics[view]) if 'semantic_cells' not in metrics[view] else metrics[view]
        old=enrich(initial[view]) if 'semantic_cells' not in initial[view] else initial[view]
        for key,a in current['semantic_cells']['learned'].items():
            d,h=map(float,key.split('/'))
            if h<settings.get('long_horizon_min',16):continue
            b=old['semantic_cells']['learned'][key];hold=current['semantic_cells']['hold'][key]
            for name in ('macro_f1','uar','neutral_recall'):
                if a[name] is None or b[name] is None:continue
                checks[f'{view}/{key}/{name}']=a[name]>=b[name]-delta
            if a['nonneutral_to_neutral'] is not None:
                checks[f'{view}/{key}/neutral_false_positive']=a['nonneutral_to_neutral']<=b['nonneutral_to_neutral']+delta
            for name in ('vad_mse','intensity_mse'):
                if a[name] is not None and b[name] is not None:checks[f'{view}/{key}/{name}']=a[name]<=b[name]*(1+relative)
            if a['macro_f1'] is not None:
                checks[f'{view}/{key}/f1_vs_hold']=a['macro_f1']>=hold['macro_f1']-delta
            details[f'{view}/{key}']=a
        # Preserve vector quality per supported domain/horizon, including short DualTalk queries.
        v=torch.tensor(current['vector_sums'])[0];prev=torch.tensor(old['vector_sums'])[0]
        for d in range(v.shape[0]):
            for h,seconds in enumerate(current['horizons']):
                if v[d,h,1]>0 and prev[d,h,1]>0:
                    checks[f'{view}/{d}/{seconds:g}/vector']=float(v[d,h,0]/v[d,h,1])<=float(prev[d,h,0]/prev[d,h,1])*(1+settings.get('vector_guard_tolerance',.05))
    checks['coverage_locked']=metrics['correction_parameters_unchanged']
    supported=[v for k,v in details.items() if k.startswith('live/') and v['macro_f1'] is not None]
    checks['long_semantics_supported']=bool(supported)
    # Existing vector-best is diagnostic; deployment must improve an actual long-horizon skill.
    gains=[]
    for key,a in metrics['live']['semantic_cells']['learned'].items():
        if float(key.split('/')[1])<settings.get('long_horizon_min',16):continue
        b=initial['live']['semantic_cells']['learned'][key]
        if a['macro_f1'] is not None:gains.append(a['macro_f1']-b['macro_f1'])
    checks['long_f1_improved']=bool(gains) and sum(gains)/len(gains)>=settings.get('minimum_long_f1_improvement',.005)
    return dict(gate_passed=all(checks.values()),checks={k:bool(v) for k,v in checks.items()},
        protocol='online-domain-horizon-semantic-neutral-and-vector-v1')


def semantic_score(metrics,settings):
    cells=metrics['live']['semantic_cells']['learned'];values=[]
    for key,value in cells.items():
        if float(key.split('/')[1])>=settings.get('long_horizon_min',16) and value['macro_f1'] is not None:
            values.append((value['macro_f1']+value['uar'])/2)
    return -sum(values)/len(values) if values else math.inf
