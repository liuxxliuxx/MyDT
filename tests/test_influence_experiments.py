"""Controlled interventions must preserve the actual causal generation path."""
import copy
import dataclasses

import pytest
import torch

from test_v2_pipeline import system, packet
from emotion_ssm.evaluate_influence import (capture_trace, replay_contexts, render_contexts,
    maximum_difference, no_event, paired_summary, score, event_trajectory, prepare, weighted)


def packets(count=13):
    values = []
    for i in range(count):
        p = packet(time=float(i+1))
        p["words"] = [{"id": f"w{i}", "role": "avatar", "text": "sad",
                       "start": i+.1, "end": i+.3, "available_at": i+.3}]
        values.append((p, torch.randn(1, 25, 56), torch.ones(1, 25, dtype=torch.bool)))
    return values


@pytest.mark.parametrize("variant", ["none", "affect", "self", "dyadic"])
def test_capture_replay_parity_and_frozen_cache(system, variant):
    system.variant = variant
    system.requires_grad_(False)
    trace = capture_trace(system, packets(5), "cpu", 5)
    context, _ = replay_contexts(system, trace)
    assert maximum_difference(context, [t["context"] for t in trace]) < 1e-6
    output = render_contexts(system, trace, context)
    assert maximum_difference(output, [t["prediction"] for t in trace]) < 1e-5
    system.generator.film.requires_grad_(True)
    with pytest.raises(ValueError, match="fully frozen"):
        capture_trace(system, packets(1), "cpu", 1)


def test_history_and_partner_intervene_only_on_state(system):
    system.requires_grad_(False)
    trace = capture_trace(system, packets(), "cpu", 13)
    donor = copy.deepcopy(trace)
    for item in donor:
        o = item["observations"][1]
        item["observations"][1] = dataclasses.replace(o, aff=-o.aff, action=o.action*2)
    for variant in ("none", "affect", "self", "dyadic"):
        system.variant = variant
        original, _ = replay_contexts(system, trace)
        reset, _ = replay_contexts(system, trace, "history_reset")
        shuffled, _ = replay_contexts(system, trace, "partner_shuffled", donor=donor)
        assert maximum_difference(original[:8], reset[:8]) == 0
        assert maximum_difference(original[:8], shuffled[:8]) == 0
        # Current affect is unchanged, including text containing original history.
        dim = 2*system.state_model.state_dim
        assert all(torch.equal(x[:, :dim], y[:, :dim]) for x, y in zip(original, shuffled))
        if variant in ("none", "affect"):
            assert maximum_difference(original, reset) == 0
        if variant != "dyadic":
            assert maximum_difference(original, shuffled) == 0
        else:
            assert maximum_difference(original[8:], shuffled[8:]) > 0
    # Replaying state does not overwrite captured target/conditioning tensors.
    assert maximum_difference([t["context"] for t in trace], replay_contexts(system, trace)[0]) == 0


def test_event_pulses_do_not_change_current_affect_or_prefix(system):
    system.requires_grad_(False)
    trace = capture_trace(system, packets(), "cpu", 13)
    event = {"index": 3, "role": "avatar"}
    base, _ = replay_contexts(system, trace, "pulse_zero", event, role="avatar")
    once, _ = replay_contexts(system, trace, "pulse_single", event, role="avatar")
    repeated, _ = replay_contexts(system, trace, "pulse_repeated", event, role="avatar")
    assert maximum_difference(base[:3], once[:3]) == 0
    assert maximum_difference(once[:5], repeated[:5]) == 0
    assert maximum_difference(base[3:], once[3:]) > 0
    for a, b in zip(base, repeated):
        assert torch.equal(a[:, :2*system.state_model.state_dim], b[:, :2*system.state_model.state_dim])
    missing = no_event(trace[3]["observations"][0])
    assert torch.count_nonzero(system.state_model.event_stimulus(missing)) == 0


def test_metrics_use_valid_elements_and_cluster_paired_rows():
    trace = [{"truth": torch.zeros(1, 25, 56), "mask": torch.ones(1, 25, dtype=torch.bool)} for _ in range(12)]
    pred = [torch.ones(1, 25, 56) for _ in trace]
    exact = [t["truth"] for t in trace]
    row = score(trace, pred, range(3, 12))
    assert row["mean_expression_mse"] == 1
    assert row["expression_mse"] == 1
    assert row["boundary_velocity_mse"] == 0
    a = {"a_speaker1": {"source": "a", "metric": row}}
    b = {"a_speaker1": {"source": "a", "metric": score(trace, exact, range(3, 12))}}
    result = paired_summary(a, b, lambda r: r["metric"], "mean_expression_mse", bootstrap=10)
    assert result["delta"] == -1
    assert result["source_cluster_95ci"] == [-1, -1]
    trajectory = event_trajectory(trace, pred, 3)
    assert trajectory["centered_trajectory_mse"] == 0  # Constant offset removed deliberately.
    assert weighted([score(trace, pred, [])]) is None


def test_selection_is_input_only_and_donors_cross_original_sources(tmp_path):
    import json
    (tmp_path/"test").mkdir()
    ids = {}
    for source in ("first", "second", "third"):
        name = source+"_sub_video_1_speaker1"
        feature = {"valid_frames": 25}
        cache = {"cache_id": source, "chunks": [{"target": feature, "partner": feature} for _ in range(20)],
            "aligned_words": [{"text": "happy", "available_at": 4.2, "role": "speaker1", "confidence": .95}]}
        torch.save(cache, tmp_path/"test"/(name+".pt"))
        ids["test/"+name] = source
    (tmp_path/"dataset_manifest.json").write_text(json.dumps({"cache_ids": ids, "digest": "declared"}))
    result = prepare(tmp_path, tmp_path/"selection.json", pairs=2)
    assert result["inventory"]["lexicon_eligible_pairs"] == 3
    assert sum(r["history"] for r in result["rows"]) == 4
    for row in result["rows"]:
        if row["donor"]:
            assert row["source"] != row["donor"].partition("_sub_video_")[0]
    # No target FLAME or generation checkpoint was needed to select any sample.
    assert all(r["event"]["index"] == 4 for r in result["rows"])
