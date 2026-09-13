"""Prepare a disposable timing run on 16 distinct training sources."""
import argparse
import copy
import json
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'source/runs/avatar_stability_20260913_gpu23/code'))
    sys.path.insert(0, str(root / 'deps'))
    from avatar_comparison_support import VerifiedRelocation, atomic_json
    from emotion_ssm.utils.checkpoint_v3 import manifest_provenance
    import torch
    torch.set_num_threads(2)
    relocation = VerifiedRelocation(root)
    payload = torch.load(relocation.export / 'current_step004000.pt', map_location='cpu', weights_only=False, mmap=True)
    config = relocation.config(payload['config'])
    initial = root / 'benchmark_initialization.pt'
    payload = dict(payload, config=config, provenance=manifest_provenance(config), metrics={}, optimizer=None,
                   scaler=None, run_state={}, rng_state=None)
    torch.save(payload, initial)
    config = copy.deepcopy(config)
    config['paths'].update(output=str(root / 'benchmark'), avatar_initialization=str(initial), resume='')
    config['train'].update(max_steps=8, validate_every=8, validation_max_dialogues=2, log_every=1,
                           experiment_scope='disposable_throughput_benchmark_16_source_subset')
    config['generation_stability'].update(retain_initial_candidate=False, diagnostic_groups={})
    atomic_json(root / 'benchmark_config.json', config)


if __name__ == '__main__':
    main()
