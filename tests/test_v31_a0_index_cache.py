"""CPU indexing must finish before NCCL and reuse only source-bound TRAIN caches."""
import copy
import itertools
import json
import threading
import time

import pytest
import torch

from emotion_ssm.data.packets_v3 import PacketFrameDataset
from emotion_ssm.models.token_observer import SUBSETS
from emotion_ssm.train import observation_v3 as training
from test_token_observer_v3 import features, model, write_cache


@pytest.fixture(autouse=True)
def one_thread():
    previous=torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def quiet(message):
    pass


def test_single_availability_scan_matches_all_seven_original_subset_checks():
    for mask in itertools.product((False,True),repeat=3):
        sample=features()
        sample["audio_mask"].fill_(mask[0])
        sample["au_mask"].fill_(mask[1]); sample["flame_mask"].fill_(False)
        sample["text_mask"].fill_(mask[2])
        expected=[name for name,subset in SUBSETS.items()
                  if bool((torch.tensor(mask)&torch.tensor(subset,dtype=torch.bool)).any())]
        assert training._available_subsets(sample)==expected


def test_source_cache_reused_across_seeds_and_calibration_without_dialogue_scan(tmp_path,monkeypatch):
    root=tmp_path/"tokens"; write_cache(root)
    dataset=PacketFrameDataset(root)
    expected=training.build_training_index([dataset])
    first=training.cached_training_index([dataset],emit=quiet)
    assert first==expected
    path,binding=training.training_index_cache_binding(dataset)
    assert path.is_file() and binding["split"]=="train"
    def unexpected(*args,**kwargs):
        pytest.fail("Repeated seed/role rank rescanned token dialogues")
    monkeypatch.setattr(training,"build_training_index",unexpected)
    for rank_zero in (True,False):
        second=training.cached_training_index([PacketFrameDataset(root)],build=rank_zero,emit=quiet)
        assert second==expected


def test_cache_rejects_stale_binding_and_detects_manifest_and_train_artifact_changes(tmp_path):
    root=tmp_path/"tokens"; write_cache(root)
    dataset=PacketFrameDataset(root)
    original=training.cached_source_training_index(dataset,emit=quiet)
    old_path,old_binding=training.training_index_cache_binding(dataset)
    data=torch.load(root/"sample.pt",weights_only=False)
    data["packets"][0]["targets"][0][0]=copy.deepcopy(data["packets"][0]["targets"][0][0])
    data["packets"][0]["targets"][0][0]["emotion"]=2
    torch.save(data,root/"sample.pt")
    new_path,new_binding=training.training_index_cache_binding(dataset)
    assert new_binding!=old_binding and new_path!=old_path
    fresh=training.cached_source_training_index(dataset,emit=quiet)
    assert original["domains"]["0"]["class_counts"][2]==0
    assert fresh["domains"]["0"]["class_counts"][2]==1
    manifest=json.loads((root/"manifest.json").read_text())
    manifest["feature_sources"]["emotiontalk"]["audio_revision"]="changed-checkpoint"
    (root/"manifest.json").write_text(json.dumps(manifest))
    latest_path,latest_binding=training.training_index_cache_binding(PacketFrameDataset(root))
    assert latest_path!=new_path and latest_binding!=new_binding
    torch.save({"binding":old_binding,"index":original},latest_path)
    with pytest.raises(ValueError,match="Stale"):
        training.cached_source_training_index(PacketFrameDataset(root),emit=quiet)


def test_index_cache_cannot_be_constructed_from_validation_split(tmp_path):
    root=tmp_path/"tokens"; write_cache(root)
    with pytest.raises(ValueError,match="train split"):
        training.cached_source_training_index(PacketFrameDataset(root,"val"),emit=quiet)


def test_cpu_file_wait_has_no_collectives_and_reads_only_atomic_ready_file(tmp_path,monkeypatch):
    root=tmp_path/"tokens"; write_cache(root)
    dataset=PacketFrameDataset(root)
    def prohibited(*args,**kwargs):
        raise AssertionError("CPU indexing must not enter a distributed collective")
    for name in ("init_process_group","broadcast_object_list","barrier","all_reduce"):
        monkeypatch.setattr(training.dist,name,prohibited)
    results=[]
    thread=threading.Thread(target=lambda:results.append(training.cached_source_training_index(
        dataset,build=False,wait_seconds=2,poll_seconds=.01,emit=quiet)))
    thread.start()
    time.sleep(.05)
    assert thread.is_alive()
    expected=training.cached_source_training_index(dataset,build=True,emit=quiet)
    thread.join(timeout=2)
    assert not thread.is_alive() and results==[expected]
    assert not list((root/".a0-training-index-v31").glob("*.tmp.*"))


def test_failed_index_publish_does_not_expose_partial_or_changed_source(tmp_path,monkeypatch):
    root=tmp_path/"tokens"; write_cache(root)
    dataset=PacketFrameDataset(root)
    path,_=training.training_index_cache_binding(dataset)
    original=training.build_training_index
    def changed(*args,**kwargs):
        index=original(*args,**kwargs)
        source=root/"sample.pt"
        source.touch()
        return index
    monkeypatch.setattr(training,"build_training_index",changed)
    with pytest.raises(ValueError,match="changed during"):
        training.cached_source_training_index(dataset,emit=quiet)
    assert not path.exists()


def test_run_initializes_distributed_only_after_cached_cpu_index_is_ready(tmp_path,monkeypatch):
    from emotion_ssm.config_v3 import default_config
    root=tmp_path/"tokens"; write_cache(root)
    config=default_config(); config["observer"]=model().construction()
    config["data"]["token_roots"]=[str(root)]
    config["train"]["device"]="cpu"
    monkeypatch.setenv("WORLD_SIZE","2"); monkeypatch.setenv("RANK","0")
    monkeypatch.setattr(training.dist,"is_initialized",lambda:False)
    class InitializedAfterIndex(Exception):
        pass
    def initialize(*args,**kwargs):
        path,binding=training.training_index_cache_binding(PacketFrameDataset(root))
        assert training._read_index_cache(path,binding)["rows"]
        raise InitializedAfterIndex
    monkeypatch.setattr(training.dist,"init_process_group",initialize)
    with pytest.raises(InitializedAfterIndex):
        training.run(config)
