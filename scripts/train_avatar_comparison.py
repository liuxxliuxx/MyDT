"""Run the archived trainer with content-verified filesystem relocation only."""
import argparse
from pathlib import Path
import runpy
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--benchmark', action='store_true')
    args, remaining = parser.parse_known_args()
    root = args.root.resolve()
    sys.path.insert(0, str(root / 'source/runs/avatar_stability_20260913_gpu23/code'))
    sys.path.insert(0, str(root / 'deps'))
    from avatar_comparison_support import VerifiedRelocation
    VerifiedRelocation(root).install()
    if args.benchmark:
        import json
        import emotion_ssm.train.generation_v3 as training
        original = training.GenerationTokenDataset
        selected = json.loads((root / 'source/runs/avatar_comparison_20260913_s2_export/smoke_selection.json').read_text())

        class BenchmarkDataset:
            def __new__(cls, token_root, raw_root, split='train'):
                data = original(token_root, raw_root, split)
                names = {name.partition(':')[2] for name in selected[split]}
                indices = [i for i, name in enumerate(data.names) if name in names]
                data.indices = [data.indices[i] for i in indices]
                data.names = [data.names[i] for i in indices]
                data.lengths = [data.lengths[i] for i in indices]
                if len(indices) != len(names):
                    raise ValueError('Benchmark dialogue selection changed')
                return data

        training.GenerationTokenDataset = BenchmarkDataset
        # Running this already imported module via runpy would replace the factory.
        sys.argv = ['emotion_ssm.train.generation_v3'] + remaining
        training.main()
        return
    sys.argv = ['emotion_ssm.train.generation_v3'] + remaining
    runpy.run_module('emotion_ssm.train.generation_v3', run_name='__main__')


if __name__ == '__main__':
    main()
