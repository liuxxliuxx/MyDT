"""Predeclared selection rules; internal visual distillation is never gold."""
from __future__ import annotations
import math


def selection_score(metrics, config=None):
    config=config or {}
    name=config.get('metric','generation_total')
    if name not in ('generation_total','generation_total_with_boundary'):
        raise ValueError('Primary selection must use a declared reconstruction metric')
    value=float(metrics[name])
    if not math.isfinite(value):
        raise ValueError('Cannot select a nonfinite metric')
    return value


def conditioned_acceptance(metrics, reference, policy, independent=None, identity=None):
    """Missing independent labels produce pending, never a semantic winner."""
    if not policy:
        return dict(status='disabled',accepted=False)
    required={'boundary_velocity_mse','jaw_mse','neck_mse','expression_mse'}
    limits=policy.get('maximum_relative_regression',{})
    if set(limits)!=required:
        raise ValueError('Declare all four deployment regression tolerances before evaluation')
    if not required<=set(reference):
        raise ValueError('Predeclare validation reference_metrics for all four deployment metrics')
    checks={k:math.isfinite(float(metrics[k])) and metrics[k]<=reference[k]*(1+float(limits[k])) for k in required}
    if any(not math.isfinite(float(v)) or v<0 for v in limits.values()):
        raise ValueError('Deployment tolerances must be finite and nonnegative')
    if independent is None:
        return dict(status='awaiting_independent_semantics',accepted=False,checks=checks)
    if (identity is None or independent.get('identity')!=identity or independent.get('split')!='val'
            or independent.get('independent_human_rows',0)<1 or not independent.get('independent_of_training_teacher')):
        raise ValueError('Independent semantics must match this checkpoint and validation population')
    metric=policy['semantic_metric']
    delta=float(independent['improvements'][metric])
    checks['independent_semantic_improvement']=math.isfinite(delta) and delta>=float(policy['minimum_improvement'])
    return dict(status='accepted' if all(checks.values()) else 'rejected',accepted=all(checks.values()),checks=checks)


def reconstruction_frontier(rows):
    axes=('expression_mse','jaw_mse','neck_mse','boundary_velocity_mse')
    valid=[row for row in rows if all(math.isfinite(float(row[k])) for k in axes)]
    def dominates(a,b):return all(a[k]<=b[k] for k in axes) and any(a[k]<b[k] for k in axes)
    return [row for row in valid if not any(dominates(other,row) for other in valid)]
