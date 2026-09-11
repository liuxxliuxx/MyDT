"""Read-only checkpoint diagnostics; stdin execution writes no remote files.

Fit post-processing coefficients on the saved training origin bank only. The
fixed validation bank is unchanged. These are checkpoint-conditional audits,
not independent model-selection or causal partner-effect experiments.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from emotion_ssm.models.state_core import UnifiedEmotionStateCore
from emotion_ssm.models.token_observer import TokenObserver
from emotion_ssm.train.dynamics_staged_v3 import fit_shrink
from emotion_ssm.train.staged_dynamics_support import (
    memory_cat, memory_index, parameter_partition, prediction_loss, weight_digest,
)


def merge(banks):
    result = dict(banks[0])
    result['states'] = memory_cat([b['states'] for b in banks])
    for name in ('targets', 'valid', 'domain'):
        result[name] = torch.cat([b[name] for b in banks])
    result['teacher_sum'] = sum(b['teacher_sum'] for b in banks)
    result['teacher_count'] = sum(b['teacher_count'] for b in banks)
    result['keys'] = sum([b['keys'] for b in banks], [])
    result['endpoints'] = []
    offset = 0
    for b in banks:
        result['endpoints'].extend((row + offset, query) for row, query in b['endpoints'])
        offset += len(b['states'].fast)
    return result


@torch.no_grad()
def predict(core, bank, device, partner=True):
    output = []
    for lo in range(0, len(bank['targets']), 128):
        state = memory_index(bank['states'], slice(lo, lo + 128)).to(device)
        output.append(torch.stack([s.z for s in core.forecast(state, bank['horizons'],
                                 enable_partner=partner)], 1).cpu())
    return torch.cat(output).double()


def metrics(prediction, bank, mean):
    truth, valid = bank['targets'].double(), bank['valid']
    n, h, _, d = truth.shape
    prediction = prediction.expand_as(truth)
    mu_direction = mean.double() / mean.double().norm()
    rows = []
    sums = torch.zeros(3, h, 2, dtype=torch.float64)
    for domain in range(3):
        for column, seconds in enumerate(bank['horizons']):
            mask = valid[:, column] & (bank['domain'] == domain)[:, None]
            if not mask.any():
                continue
            p, y = prediction[:, column][mask], truth[:, column][mask]
            error = p - y
            mse = error.square().mean()
            bias = error.mean(0).square().mean()
            sums[domain, column] = torch.tensor([error.square().sum(), error.numel()])
            rows.append(dict(domain=domain, seconds=seconds, queries=len(y), mse=float(mse),
                mean_error_bias_mse=float(bias), centered_error_mse=float(mse-bias),
                mean_axis_error_mse=float((error @ mu_direction).square().mean()/d),
                prediction_norm=float(p.norm(dim=-1).mean()),
                target_norm=float(y.norm(dim=-1).mean()),
                cosine=float(F.cosine_similarity(p, y, dim=-1).mean())))
    error, count = sums[..., 0], sums[..., 1]
    per_h = [(error[count[:, i] > 0, i]/count[count[:, i] > 0, i]).mean() for i in range(h)]
    return dict(mse=float(error.sum()/count.sum()), macro_mse=float(torch.stack(per_h).mean()),
                elements=int(count.sum()), rows=rows)


def fit_processing(pred, bank, mean):
    origin, truth, valid = bank['states'].z.double(), bank['targets'].double(), bank['valid']
    alpha, biases, beta = [], [], []
    for column in range(len(bank['horizons'])):
        mask = valid[:, column]
        s, p, y = origin[mask], pred[:, column][mask], truth[:, column][mask]
        delta = p-s
        alpha.append(((y-s)*delta).sum()/delta.square().sum().clamp_min(1e-12))
        biases.append((y-p).mean(0))
        # Scalar shift toward the training mean from the frozen core baseline.
        direction = mean.double()-bank['states'].baseline[0, 0].double()
        beta.append(((y-p)*direction).sum()/(len(y)*direction.square().sum()).clamp_min(1e-12))
    return torch.stack(alpha).clamp(0, 1), torch.stack(biases), torch.stack(beta)


def representation(bank, mean, baseline):
    # Query-weighted diagnostics; unlike fit_shrink's mean, targets repeat at
    # different origins/horizons. Report this distinction explicitly.
    values = bank['targets'].double()[bank['valid']]
    n = len(values)
    center = values.mean(0)
    centered = values-center
    eigenvalues = torch.linalg.eigvalsh(centered.T @ centered/n).clamp_min(0)
    return dict(valid_target_queries=n, dimension=values.shape[-1],
        query_weighted_mean_norm=float(center.norm()),
        target_coordinate_variance_mean=float(centered.square().mean()),
        effective_covariance_rank=float(eigenvalues.sum().square()/eigenvalues.square().sum()),
        top_eigenvalue_variance_share=float(eigenvalues[-1]/eigenvalues.sum()),
        mean_target_cosine_distinct_pairs=float((values.sum(0).square().sum()-values.square().sum())/(n*(n-1))),
        train_mean_norm=float(mean.norm()), baseline_norm=float(baseline.norm()),
        mean_baseline_rms=float((mean-baseline).square().mean().sqrt()),
        mean_baseline_cosine=float(F.cosine_similarity(mean, baseline, dim=0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True)
    args = parser.parse_args()
    root = Path(args.run)/'formal'
    device = torch.device('cuda:0')
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    checkpoint = torch.load(root/'best.pt', map_location='cpu', weights_only=False)
    block = checkpoint['run_state']['block']
    core = UnifiedEmotionStateCore.from_config(checkpoint['construction']['state']).to(device).eval()
    observer = TokenObserver(checkpoint['construction']['observer']).to(device).eval()
    core.load_state_dict(checkpoint['models']['state'], strict=True)
    observer.load_state_dict(checkpoint['models']['observer'], strict=True)
    core.configure_execution('optimized')
    core.requires_grad_(False); observer.requires_grad_(False)
    before = weight_digest({'state': core, 'observer': observer})
    train = merge([torch.load(root/f'origins_block{block}_rank{r}.pt', map_location='cpu',
                             weights_only=False) for r in (0, 1)])
    val = merge([torch.load(root/f'reference_val_rank{r}.pt', map_location='cpu',
                           weights_only=False) for r in (0, 1)])
    mean, shrink = fit_shrink(train, device)
    mean, shrink = mean.cpu().double(), shrink.cpu().double()
    report = dict(time=datetime.datetime.now().astimezone().isoformat(), step=checkpoint['global_step'],
        candidate_hash=before, block=block, protocol='train_fitted_postprocessing_fixed_validation_v1',
        train_origins=len(train['targets']), validation_origins=len(val['targets']),
        train_dialogues=len(set(str(k[0]) for k in train['keys'])),
        validation_dialogues=len(set(str(k[0]) for k in val['keys'])),
        shrink_coefficients=shrink.tolist(), horizons=train['horizons'],
        teacher_count=train['teacher_count'], protocols={}, training_gradients=[])
    print('predicting training bank', file=sys.stderr, flush=True)
    train_prediction = predict(core, train, device)
    alpha, biases, beta = fit_processing(train_prediction, train, mean)
    report['train_fitted_alpha'] = alpha.tolist()
    report['train_fitted_center_shift_beta'] = beta.tolist()
    report['train_fitted_bias_norm'] = biases.norm(dim=-1).tolist()
    report['teacher_mean'] = mean.tolist()
    report['baseline'] = core.baseline.detach().cpu().tolist()
    for name, bank, p in [('train', train, train_prediction), ('fixed_val', val, None)]:
        print('evaluating '+name, file=sys.stderr, flush=True)
        if p is None:
            p = predict(core, bank, device)
        s = bank['states'].z.double()[:, None]
        b = mean+shrink[None, :, None, None]*(s-mean)
        methods = dict(learned=p, hold=s, shrink=b, mean=mean[None, None, None],
            learned_alpha=s+alpha[None, :, None, None]*(p-s),
            learned_train_bias=p+biases[None, :, None],
            learned_center_shift=p+beta[None, :, None, None]*(mean-bank['states'].baseline[0, 0].double()),
            normalized_learned=F.normalize(p, dim=-1))
        if name == 'fixed_val':
            methods['without_future_partner'] = predict(core, bank, device, partner=False)
        results = {method: metrics(value, bank, mean) for method, value in methods.items()}
        if name == 'fixed_val':
            for method in ('learned', 'hold', 'shrink', 'mean'):
                expected = checkpoint['metrics']['fixed']['methods'][method]['macro_mse']
                delta = abs(results[method]['macro_mse']-expected)
                if delta > 2e-8:
                    raise AssertionError(f'{method} metric replay mismatch {delta}')
            # Paired dialogue wins; no claim that overlapping query elements
            # are independent statistical observations.
            unique = sorted(set(str(k[0]) for k in bank['keys']))
            wins = {}
            for method in ('learned', 'learned_train_bias', 'without_future_partner'):
                wins[method] = 0
                error = ((methods[method]-bank['targets'].double()).square().mean(-1)
                         -(b-bank['targets'].double()).square().mean(-1))
                for key in unique:
                    rows = torch.tensor([str(k[0]) == key for k in bank['keys']])
                    mask = bank['valid'] & rows[:, None, None]
                    wins[method] += int(error[mask].mean() < 0)
            report['dialogue_wins_over_shrink'] = dict(wins=wins, total=len(unique))
        report['protocols'][name] = dict(methods=results,
            representation=representation(bank, mean, core.baseline.detach().cpu().double()),
            origins_per_domain=torch.bincount(bank['domain'], minlength=3).tolist(),
            endpoint_queries=len(bank['endpoints']))
        print(json.dumps({name:{k:v['macro_mse'] for k,v in results.items()}}), file=sys.stderr, flush=True)
    parameters = parameter_partition(observer, core, 'readapt')
    generator = torch.Generator().manual_seed(8675309)
    for domain in range(3):
        pool = torch.where(train['domain'] == domain)[0]
        for repeat in range(2):
            indices = pool[torch.randint(len(pool), (16,), generator=generator)]
            coordinate, _ = prediction_loss(core, train, indices, device)
            grads_c = torch.autograd.grad(coordinate, parameters, allow_unused=True)
            total, _ = prediction_loss(core, train, indices, device, observer, 1.)
            grads_t = torch.autograd.grad(total, parameters, allow_unused=True)
            gc = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                            for p,g in zip(parameters,grads_c)])
            gt = torch.cat([(torch.zeros_like(p) if g is None else g).flatten()
                            for p,g in zip(parameters,grads_t)])
            gl = gt-gc
            row = dict(domain=domain, repeat=repeat, coordinate_loss=float(coordinate),
                label_loss=float(total-coordinate), coordinate_gradient_norm=float(gc.norm()),
                label_gradient_norm=float(gl.norm()),
                label_coordinate_gradient_cosine=float(F.cosine_similarity(gc, gl, dim=0)),
                label_to_coordinate_gradient_ratio=float(gl.norm()/gc.norm().clamp_min(1e-12)))
            report['training_gradients'].append(row)
            print(json.dumps(row), file=sys.stderr, flush=True)
    after = weight_digest({'state': core, 'observer': observer})
    if before != after:
        raise AssertionError('Diagnostic modified model weights')
    report['weights_unchanged'] = True
    report['completed_at'] = datetime.datetime.now().astimezone().isoformat()
    print(json.dumps(report, ensure_ascii=False, allow_nan=False))


if __name__ == '__main__':
    main()
