"""One-batch CPU prefetch with a separate, committed training cursor.

Only explicitly RNG-neutral datasets may use this thread. No model, CUDA call,
augmentation or training RNG runs in the worker. Speculative reads are never
included in packets_seen, so checkpoint restoration reads the same next packet.
"""
from concurrent.futures import ThreadPoolExecutor
import time


class PrefetchSegmentCursor:
    def __init__(self, cursor, count, lookahead=0):
        if not getattr(cursor.dataset, "prefetch_rng_neutral", False):
            raise ValueError("Prefetch requires an explicitly RNG-neutral CPU dataset")
        if count < 1 or lookahead < 0:
            raise ValueError("Invalid prefetch packet budget")
        self.cursor, self.count, self.lookahead = cursor, int(count), int(lookahead)
        self.seen = cursor.seen
        self._committed_state = cursor.state_dict() if hasattr(cursor, "state_dict") else None
        self._peek = []
        self.last_load_seconds = 0.
        self._closed = False
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-data")
        self._future = self._pool.submit(self._read)

    def _read(self):
        began = time.perf_counter()
        packets = self.cursor.take_valid(self.count)
        future = self.cursor.peek(self.lookahead) if self.lookahead else []
        saved = self.cursor.state_dict() if hasattr(self.cursor, "state_dict") else None
        return packets, future, self.cursor.seen, time.perf_counter() - began, saved

    def take_valid(self, count):
        if self._closed or count != self.count:
            raise ValueError("Prefetch uses a fixed valid-block budget until closed")
        packets, self._peek, seen, self.last_load_seconds, self._committed_state = self._future.result()
        self.seen = seen
        self._future = self._pool.submit(self._read)
        return packets

    def state_dict(self):
        import copy
        return copy.deepcopy(self._committed_state)

    def peek(self, count):
        if count < 0 or count > self.lookahead:
            raise ValueError("Lookahead exceeds the declared prefetch label horizon")
        return list(self._peek[:count])

    def close(self):
        if not self._closed:
            self._closed = True
            self._future.cancel()
            self._pool.shutdown(wait=True, cancel_futures=True)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
