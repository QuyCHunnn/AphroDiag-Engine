"""Experimental request coalescing for the row-batched acoustic graph.

Each cue retains SDK normalization, global decoding, codec and babble guard.
Only acoustic frame requests are coalesced. No vendor files are modified.
"""
from concurrent.futures import Future
from copy import copy
import queue
import math
import threading
import time
from types import MethodType


class AcousticBatcher:
    def __init__(self, runtime, max_wait_seconds=.012):
        if not math.isfinite(max_wait_seconds) or max_wait_seconds < 0:
            raise ValueError("Batch wait must be finite and non-negative")
        self.runtime = runtime
        self.max_wait = max_wait_seconds
        self.queue = queue.Queue(maxsize=runtime.batch_size*2)
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._work, daemon=True)
        self.thread.start()

    def submit(self, tokens, history, uniforms):
        if self.error is not None:
            raise RuntimeError("Acoustic batch worker failed") from self.error
        if self.stopping.is_set():
            raise RuntimeError("Acoustic batch worker is closed")
        future = Future()
        self.queue.put((tokens, history, uniforms, future), timeout=30)
        if (self.error is not None or self.stopping.is_set()) and not future.done():
            future.set_exception(self.error or RuntimeError("Acoustic batch worker closed"))
        return future.result(timeout=120)

    def _work(self):
        pending = []
        try:
            while not self.stopping.is_set():
                try:
                    item = self.queue.get(timeout=.05)
                except queue.Empty:
                    continue
                pending = [item]
                deadline = time.monotonic()+self.max_wait
                while len(pending) < self.runtime.batch_size:
                    remaining = deadline-time.monotonic()
                    if remaining <= 0:
                        break
                    try:
                        pending.append(self.queue.get(timeout=remaining))
                    except queue.Empty:
                        break
                codes, eos = self.runtime.run([p[0] for p in pending], [p[1] for p in pending], [p[2] for p in pending])
                if len(codes) != len(pending) or len(eos) != len(pending):
                    raise RuntimeError("Acoustic batch returned missing rows")
                for row, request in enumerate(pending):
                    if not request[3].done():
                        request[3].set_result((codes[row], bool(eos[row])))
                pending = []
        except BaseException as exc:
            self.error = exc
        finally:
            failure = self.error or RuntimeError("Acoustic batch worker closed")
            for request in pending:
                if not request[3].done():
                    request[3].set_exception(failure)
            while True:
                try:
                    request = self.queue.get_nowait()
                except queue.Empty:
                    break
                if not request[3].done():
                    request[3].set_exception(failure)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("Acoustic runtime did not finish during shutdown")


def clone_model_for_request(model, seed, batcher=None):
    """Independent generation state/RNG; immutable weights and ORT sessions shared.

    ORT sessions are reentrant. The experimental GPU InferRequest is used only
    by AcousticBatcher's single service thread, never concurrently.
    """
    import numpy as np
    cloned = copy(model)
    cloned.engine = copy(model.engine)
    cloned.engine._lock = threading.RLock()
    rng = np.random.RandomState(seed)

    def sample(self, logits, temperature, top_k, top_p, penalty, previous):
        # Same sampling policy as the installed Apache-2.0 VieNeu ONNX SDK,
        # replacing only the process-global RNG with a request-local RNG.
        values = logits.astype(np.float32)
        if penalty != 1.0 and previous:
            indices = np.fromiter(previous, dtype=np.int64, count=len(previous))
            selected = values[indices]
            values = values.copy()
            values[indices] = np.where(selected < 0, selected*penalty, selected/penalty)
        if not temperature or temperature <= 0:
            return int(values.argmax())
        values = values/temperature
        candidates = np.argpartition(values, -int(top_k))[-int(top_k):] if top_k and 0 < int(top_k) < len(values) else np.arange(len(values))
        ordered = np.argsort(values[candidates])[::-1]
        candidates = candidates[ordered]
        probabilities = np.exp(values[candidates]-np.max(values[candidates]))
        probabilities /= probabilities.sum()
        if top_p and top_p < 1:
            probabilities *= (np.cumsum(probabilities)-probabilities) < top_p
            probabilities /= probabilities.sum()
        return int(candidates[rng.choice(len(candidates), p=probabilities)])

    cloned.engine._sample = MethodType(sample, cloned.engine)
    if batcher is not None:
        def acoustic(self, hidden, temperature, top_k, top_p, penalty, hist):
            if (temperature, top_k, top_p, penalty) != (.8, 25, .95, 1.2):
                raise ValueError("Experimental batched graph supports SDK default sampling only")
            history = np.zeros(self.audio_emb.shape[:2], bool)
            if hist is not None:
                for channel in range(self.n_vq):
                    history[channel, list(hist[channel])] = True
            tokens = np.stack([hidden[0].astype(np.float32), self.text_emb[self.sgs].astype(np.float32)])
            codes, eos = batcher.submit(tokens, history, rng.random(self.n_vq).astype(np.float32))
            codes = [int(c) for c in codes]
            if hist is not None:
                for channel, code in enumerate(codes):
                    hist[channel].add(code)
            return codes, eos
        cloned.engine._acoustic_frame = MethodType(acoustic, cloned.engine)
    return cloned
