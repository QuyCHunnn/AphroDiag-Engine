"""Opt-in cross-process benchmark spans; unset JELLYDUB_TRACE_FILE costs no I/O."""
from contextlib import contextmanager
import json
import os
import time
import uuid


def _event(path, **values):
    data = (json.dumps({"pid": os.getpid(), "monotonic_seconds": time.monotonic(), **values})+"\n").encode()
    # A separate open/close with O_APPEND keeps worker records independent;
    # this diagnostic stream never decides whether pipeline work succeeds.
    try:
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.write(fd, data)
        finally:
            os.close(fd)
    except OSError:
        pass


@contextmanager
def span(phase, **metadata):
    path = os.environ.get("JELLYDUB_TRACE_FILE")
    if not path:
        yield
        return
    identity = uuid.uuid4().hex
    _event(path, event="begin", span=identity, phase=phase, metadata=metadata)
    started = time.monotonic()
    status = "passed"
    try:
        yield
    except BaseException:
        status = "failed"
        raise
    finally:
        _event(path, event="end", span=identity, phase=phase,
               seconds=time.monotonic()-started, status=status)
