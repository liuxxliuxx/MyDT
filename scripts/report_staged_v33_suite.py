"""Incremental source-backed curves for every v3.3 validation checkpoint."""
import argparse
import io
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def check_environment():
    """Exercise the headless PNG renderer before any expensive training starts."""
    fig,ax=plt.subplots()
    try:
        ax.plot([0,1],[0,1]);ax.set_xlabel('Optimizer steps')
        with io.BytesIO() as buffer:
            fig.savefig(buffer,format='png')
            if not buffer.getvalue().startswith(b'\x89PNG\r\n\x1a\n'):
                raise RuntimeError('Report renderer did not produce a PNG')
        return dict(status='passed',matplotlib=matplotlib.__version__,backend=matplotlib.get_backend())
    finally:plt.close(fig)


def report(root):
    root=Path(root);lines=['# v3.3 controlled experiments','',
        'All values below use the new query protocol. These are validation results, not independent test results.','',
        '| Experiment | Step | Fixed MSE | Live MSE | Live shrink MSE | DualTalk MSE | Endpoint F1 | Deploy gate |',
        '|---|---:|---:|---:|---:|---:|---:|---|']
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        paths=sorted(folder.glob('validation_step*.json'))
        if not paths:continue
        values=[json.loads(p.read_text()) for p in paths];last=values[-1];steps=[v['step'] for v in values]
        fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
        for ax,protocol in zip(axes.flat,['fixed','live','update_gap','sensor_gap']):
            for method in ('learned','hold','decay','mean','shrink','continuous_shrink','calibrated_hold','current_linear','history_linear','center_amplitude'):
                ax.plot(steps,[v[protocol]['methods'][method]['macro_mse'] for v in values],label=method,linewidth=1.2)
            ax.set(title=protocol,xlabel='Optimizer steps',ylabel='Domain/horizon macro vector MSE');ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,ncol=2);fig.suptitle(folder.name);fig.savefig(folder/'mse_curves.png',dpi=160);plt.close(fig)
        m=last['live']['methods'];domain=m['learned']['by_domain'];semantic=last['live']['semantic']['learned']['endpoint_balanced']
        def number(x):return '-' if x is None else f'{x:.7g}'
        lines.append('| '+' | '.join([folder.name,str(last['step']),number(last['fixed']['methods']['learned']['macro_mse']),
            number(m['learned']['macro_mse']),number(m['shrink']['macro_mse']),number(domain[2] if len(domain)>2 else None),
            number(semantic['macro_f1']),str(last['acceptance']['gate_passed'])])+' |')
    (root/'comparison.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


if __name__=='__main__':
    p=argparse.ArgumentParser();group=p.add_mutually_exclusive_group(required=True)
    group.add_argument('--root');group.add_argument('--check-environment',action='store_true');args=p.parse_args()
    if args.check_environment:print(json.dumps(check_environment()))
    else:report(args.root)
