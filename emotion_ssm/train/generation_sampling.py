"""Interleave chronological conversations without dropping their persistent state."""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field

from emotion_ssm.models.streaming_v3 import map_tensors

SAMPLING_PROTOCOL = "source-sharded-interleaved-conversations-v1"


def source_id(name):
    # This is the dataset's original-video naming convention, not a claim that
    # numbered subclips are temporally adjacent. State never crosses files.
    return str(name).removeprefix("dualtalk:").split("_sub_video_", 1)[0]


def session_key(packet):
    return str(packet["session_id"]), tuple(packet["roles"])


@dataclass
class ConversationStates:
    states: dict = field(default_factory=dict)
    previous: dict = field(default_factory=dict)
    boundary_for_loss: dict = field(default_factory=dict)

    def __setstate__(self,values):
        self.__dict__.update(values)
        if 'boundary_for_loss' not in values:
            from emotion_ssm.utils.generation_losses import BoundaryFrame
            self.boundary_for_loss={}
            for key,edge in self.previous.items():
                state=self.states.get(key)
                if state is not None and len(edge)==3:
                    self.boundary_for_loss[key]=BoundaryFrame(state.session_id,state.roles,round(state.time*25),*edge)

    def detach(self):
        return map_tensors(self, lambda value: value.detach())

    def to(self, device):
        return map_tensors(self, lambda value: value.to(device))


class InterleavedCursor:
    """Exactly-once records per epoch, fixed source ownership, compact resume.

    A slot contributes `blocks_per_conversation` valid targets at each step.
    Missing-target packets still advance observations. Each slot runs its own
    dialogue chronologically; completed records are replaced without resetting
    any other slot. Iterators and speculative prefetched data are not serialized.
    """

    def __init__(self, dataset, seed=6666, rank=0, world_size=1,
                 conversations=4, blocks_per_conversation=4, saved=None):
        self.dataset, self.seed = dataset, int(seed)
        self.rank, self.world_size = int(rank), int(world_size)
        self.conversations, self.blocks = int(conversations), int(blocks_per_conversation)
        if min(self.conversations, self.blocks, self.world_size) < 1:
            raise ValueError("Invalid interleaved sampling budget")
        self.sources = list(getattr(dataset,'source_ids',[source_id(name) for name in dataset.names]))
        sources = sorted(set(self.sources))
        random.Random(self.seed).shuffle(sources)
        weights = {source: 0 for source in sources}
        for index, source in enumerate(self.sources):
            weights[source] += int(dataset.lengths[index])
        # Static greedy assignment balances actual optimization packets while
        # keeping every original video's records on one rank for all epochs.
        sources.sort(key=lambda source: weights[source], reverse=True)
        loads, ownership = [0] * self.world_size, {}
        for source in sources:
            owner = min(range(self.world_size), key=lambda rank: loads[rank])
            ownership[source] = owner
            loads[owner] += weights[source]
        owned = {source for source, owner in ownership.items() if owner == self.rank}
        self.shard_packet_counts = loads
        self.indices = [i for i, source in enumerate(self.sources) if source in owned]
        if len(owned) < self.conversations:
            raise ValueError("Not enough independent source videos for conversation slots")
        if any(int(dataset.lengths[i]) <= 0 for i in self.indices):
            raise ValueError("Empty conversations must be removed before generation sampling")
        self.epoch, self.seen = 0, 0
        self.pending = self._order()
        self.slots = [None] * self.conversations
        self.iterators = [None] * self.conversations
        if saved is not None:
            expected = (SAMPLING_PROTOCOL, self.seed, self.rank, self.world_size, self.conversations, self.blocks)
            actual = tuple(saved[k] for k in ("protocol", "seed", "rank", "world_size", "conversations", "blocks"))
            if actual != expected:
                raise ValueError("Interleaved sampling protocol changed on resume")
            self.epoch, self.seen = int(saved["epoch"]), int(saved["seen"])
            self.pending, self.slots = copy.deepcopy(saved["pending"]), copy.deepcopy(saved["slots"])
            if len(self.slots) != self.conversations:
                raise ValueError("Invalid saved conversation slots")
            for slot in self.slots:
                if slot is not None and slot["index"] not in self.indices:
                    raise ValueError("Saved conversation belongs to a different source shard")

    def _order(self):
        order = list(self.indices)
        random.Random(self.seed + self.epoch).shuffle(order)
        return order

    def state_dict(self):
        return copy.deepcopy(dict(protocol=SAMPLING_PROTOCOL, seed=self.seed, rank=self.rank,
            world_size=self.world_size, conversations=self.conversations, blocks=self.blocks,
            epoch=self.epoch, seen=self.seen, pending=self.pending, slots=self.slots))

    def _fill(self, slot_index):
        if not self.pending:
            self.epoch += 1
            self.pending = self._order()
        occupied = {self.sources[slot["index"]] for slot in self.slots if slot is not None}
        # Prefer distinct videos. At an epoch tail all remaining records can
        # share a source; consume them without duplication and report coverage.
        offset = next((j for j, index in enumerate(self.pending) if self.sources[index] not in occupied), 0)
        index = self.pending.pop(offset)
        self.slots[slot_index] = dict(index=index, position=0, epoch=self.epoch)

    def _next(self, slot_index):
        if self.slots[slot_index] is None:
            self._fill(slot_index)
        slot = self.slots[slot_index]
        if self.iterators[slot_index] is None:
            iterator = iter(self.dataset.packets(slot["index"]))
            for _ in range(slot["position"]):
                next(iterator)
            self.iterators[slot_index] = iterator
        try:
            packet, target, valid = next(self.iterators[slot_index])
        except StopIteration as error:
            raise ValueError("Manifest packet length differs from actual conversation") from error
        packet = dict(packet)
        packet["session_id"] = f"{slot['epoch']}:{packet['session_id']}"
        packet["training_source_id"] = self.sources[slot["index"]]
        packet["training_record_name"] = self.dataset.names[slot["index"]]
        slot["position"] += 1
        self.seen += 1
        packet["training_session_end"] = slot["position"] == int(self.dataset.lengths[slot["index"]])
        if packet["training_session_end"]:
            self.slots[slot_index], self.iterators[slot_index] = None, None
        return packet, target, valid

    def take_valid(self, count, max_packets=10000):
        if count != self.conversations * self.blocks:
            raise ValueError("Per-rank block budget must equal conversations times blocks per conversation")
        packets = []
        for slot in range(self.conversations):
            valid_count = 0
            while valid_count < self.blocks:
                item = self._next(slot)
                packets.append(item)
                valid_count += bool(item[2].any())
                if len(packets) >= max_packets:
                    raise ValueError("Insufficient valid targets in interleaved source shard")
        return packets

    def peek(self, count):
        if count:
            raise ValueError("Interleaved joint future labels require per-session lookahead; unsupported")
        return []
