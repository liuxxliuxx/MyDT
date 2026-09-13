"""Evaluate fixed current/official/fine-tuned weights through one 25-frame renderer."""
from __future__ import annotations
import argparse
import copy
import json
import os
from pathlib import Path
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--model', choices=('current4000', 'official', 'baseline_control'), required=True)
    parser.add_argument('--split', choices=('val', 'test', 'ood'), default='val')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    root = args.root.resolve()
    code = root / 'source/runs/avatar_stability_20260913_gpu23/code'
    sys.path.insert(0, str(code))
    sys.path.insert(0, str(root / 'deps'))
    from avatar_comparison_support import VerifiedRelocation, atomic_json, identity, sha256
    import torch
    from emotion_ssm.utils.checkpoint_v3 import build_avatar, load_avatar
    from emotion_ssm.utils.distributed import init_distributed
    from emotion_ssm.train.generation_v3 import GenerationTokenDataset, evaluate
    torch.set_num_threads(2)
    relocation = VerifiedRelocation(root)
    relocation.install()
    cfg = json.loads((relocation.export / 'source_config.json').read_text())
    cfg = relocation.config(cfg)
    context = init_distributed('cuda:0', 6666, True)
    started = time.time()
    if args.model == 'current4000':
        path = relocation.export / 'current_step004000.pt'
        relocation.token_path(path)
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        model, _, _ = load_avatar(payload, context.device)
        source_step = payload['global_step']
        del payload
    else:
        construction = json.loads((relocation.export / 'source_construction.json').read_text())
        construction['features'] = None
        cfg['generation']['variant'] = 'none'
        model = build_avatar(cfg, device='cpu', construction=construction, initialize=False)
        path = (relocation.project / 'model/dualtalk_baseline.pth' if args.model == 'official' else
                relocation.project / 'runs/dualtalk_baseline_control/equal_epoch3/baseline_control_best.pt')
        payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
        model.generator.load_baseline_state_dict(payload, strict=True)
        source_step = 0 if args.model == 'official' else payload.get('global_step')
        del payload
        model.to(context.device)
    if model.features is not None:
        model.features.to('cpu')
    model.state_model.configure_execution('optimized')
    model.requires_grad_(False).eval()
    data = GenerationTokenDataset(cfg['data']['dualtalk_tokens'], cfg['data']['dualtalk_raw'], args.split)
    progress_path = root / 'progress' / (args.model + '_' + args.split + ('_smoke' if args.smoke else '') + '.json')
    original_packets = data.packets
    progress = dict(model=args.model, split=args.split, smoke=args.smoke, chunks=0, completed_dialogues=0,
                    total_dialogues=min(16,len(data)) if args.smoke else len(data), started=time.time())

    def timed_packets(index):
        for item in original_packets(index):
            yield item
            progress['chunks'] += 1
            if progress['chunks'] % 200 == 0:
                atomic_json(progress_path, dict(progress, updated=time.time()))
        progress['completed_dialogues'] += 1
        atomic_json(progress_path, dict(progress, updated=time.time()))

    data.packets = timed_packets
    # FP32 matches training validation. No test/OOD input or metric affects selection.
    start_eval = time.time()
    metrics, records = evaluate(model, data, context.device, max_dialogues=16 if args.smoke else 0,
                                representation_diagnostics=False)
    torch.cuda.synchronize()
    elapsed = time.time() - start_eval
    import transformers
    result = dict(protocol='avatar-unified-25-frame-comparison-v1', model=args.model,
        source_checkpoint=str(path), source_step=source_step, split=args.split,
        complete_evaluation=not args.smoke, metrics=metrics, dialogues=records,
        identity=identity(records), manifest_digest=data.manifest_digest,
        input_protocol=dict(new_frames=25, previous_context_seconds=3, chronological=True,
                            target_flame_observed=False, precision='float32'),
        runtime=dict(device=torch.cuda.get_device_name(0), torch=torch.__version__,
                     transformers=transformers.__version__, evaluation_seconds=elapsed,
                     total_seconds=time.time()-started, chunks_per_second=metrics['evaluated_chunks']/elapsed),
        source_manifest_sha256=sha256(relocation.export / 'transfer_manifest.json'))
    folder = root / ('smoke' if args.smoke else 'evaluation')
    atomic_json(folder / (args.model + '_' + args.split + '.json'), result)
    print(json.dumps(dict(model=args.model, split=args.split, metrics=metrics, runtime=result['runtime'])), flush=True)


if __name__ == '__main__':
    main()
