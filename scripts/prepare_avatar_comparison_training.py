"""Prepare matched fresh controls; official backbone, frozen old9750 emotion system."""
from __future__ import annotations
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
    import torch
    from avatar_comparison_support import VerifiedRelocation, atomic_json, sha256
    from emotion_ssm.utils.checkpoint_v3 import build_avatar, manifest_provenance
    from emotion_ssm.train.staged_dynamics_support import weight_digest
    relocation = VerifiedRelocation(root)
    torch.set_num_threads(2)
    payload = torch.load(relocation.export / 'current_step004000.pt', map_location='cpu', weights_only=False, mmap=True)
    cfg = relocation.config(payload['config'])
    construction = copy.deepcopy(payload['construction'])
    construction['features'] = None
    model = build_avatar(cfg, device='cpu', construction=construction, initialize=False)
    official = relocation.project / 'model/dualtalk_baseline.pth'
    model.generator.load_baseline_state_dict(torch.load(official, map_location='cpu', weights_only=False, mmap=True), strict=True)
    system = dict(payload['models']['system'])
    for key in list(system):
        if key.startswith('generator.baseline.'):
            del system[key]
    system.update({'generator.baseline.' + key: value for key, value in model.generator.baseline.state_dict().items()})
    for key in system:
        if key.startswith('generator.film.'):
            system[key] = torch.zeros_like(system[key])
    condition = {name: {key[len(prefix):]: value for key, value in system.items() if key.startswith(prefix)}
                 for name, prefix in [('observer', 'observer.'), ('teacher', 'teacher.'), ('state', 'state_model.')]}
    condition_hash = weight_digest(condition)
    expected = 'b1f656f069b3b86cdb18f9c59021d84746b606c84a93dd33d5e76d1283db860e'
    if condition_hash != expected:
        raise ValueError('Frozen observer/teacher/old9750 dynamics changed')
    actual_provenance = manifest_provenance(cfg)
    relocated_provenance = {str(relocation.relocate(path)): metadata for path, metadata in payload['provenance'].items()}
    if actual_provenance != relocated_provenance:
        raise ValueError('Relocated manifests differ from original checkpoint')
    output = root / 'initialization'
    output.mkdir(exist_ok=True)
    jobs = []
    for variant in ('none', 'affect', 'self', 'dyadic'):
        initial = output / (variant + '.pt')
        variant_cfg = copy.deepcopy(cfg)
        variant_cfg['generation']['variant'] = variant
        variant_cfg['paths'].update(resume='', avatar_initialization=str(initial), baseline=str(official),
                                    dynamics_checkpoint='', observation_checkpoint='')
        variant_cfg['generation_stability']['retain_initial_candidate'] = False
        variant_cfg['train']['experiment_scope'] = 'matched_official_init_frozen_old9750_avatar_controls'
        saved = dict(payload, config=variant_cfg, models={'system': system}, optimizer=None, scaler=None,
                     run_state={}, rng_state=None, metrics={}, global_step=0, provenance=actual_provenance,
                     initialization_protocol=dict(name='matched-official-backbone-zero-film-v1',
                         source_avatar_step=4000, retained_from_avatar='frozen observer/teacher/dynamics/extractor only',
                         generator_source=str(official), frozen_condition_hash=condition_hash,
                         film='zero identity initialization', old_optimizer_restored=False,
                         original_provenance=payload['provenance']))
        if initial.exists():
            raise ValueError('Refusing to overwrite an existing initialization: ' + str(initial))
        torch.save(saved, initial.with_suffix('.tmp'))
        initial.with_suffix('.tmp').replace(initial)
        for seed in (6666, 6667, 6668):
            config = copy.deepcopy(variant_cfg)
            config['train']['seed'] = seed
            config['paths']['output'] = str(root / 'training' / ('seed' + str(seed)) / variant)
            path = root / 'configs' / (variant + '_seed' + str(seed) + '.json')
            atomic_json(path, config)
            jobs.append(dict(variant=variant, seed=seed, config=str(path), output=config['paths']['output'],
                             pilot_steps=1000, max_steps=30000, global_valid_blocks=32))
    atomic_json(root / 'training_plan.json', dict(protocol='matched-official-backbone-zero-film-v1',
        condition_hash=condition_hash, generator_sha256=sha256(official),
        shared_initialization=True, variants=['none', 'affect', 'self', 'dyadic'], seeds=[6666, 6667, 6668],
        primary_selection='full val generation_total', test_ood_selection=False,
        diagnostic_steps_per_group=1000, optimizer_steps_per_group=30000,
        global_valid_blocks=32, jobs=jobs,
        comparison_to_server1='Server1 is a warm-started historical candidate; these four groups form the matched ablation.'))
    print(json.dumps(dict(status='ready', trials=len(jobs), condition_hash=condition_hash)), flush=True)


if __name__ == '__main__':
    main()
