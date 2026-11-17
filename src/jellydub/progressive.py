"""Bounded IPC prefetch while native ASR produces completed regions."""
from dataclasses import asdict
import json
import os
from pathlib import Path
from .performance_trace import span
import subprocess
import sys
import signal
import tempfile


class ProgressivePrefetch:
    def __init__(self, config, report):
        config.cache_dir.mkdir(parents=True, exist_ok=True)
        self.report = report
        self.directory = tempfile.TemporaryDirectory(prefix="prefetch-", dir=config.cache_dir)
        self.log = (Path(self.directory.name)/"worker.log").open("w+")
        options = {"stdin": subprocess.PIPE, "stdout": self.log, "stderr": subprocess.STDOUT, "text": True,
                   "start_new_session": True}
        if sys.platform.startswith("linux"):
            options["pipesize"] = 4096
        self.failed = False
        self.process = None
        try:
            self.process = subprocess.Popen([sys.executable, str(Path(__file__).with_name("progressive_worker.py"))], **options)
            self.process.stdin.write(json.dumps(asdict(config), default=str)+"\n")
            self.process.stdin.flush()
        except BaseException:
            self.close()
            raise

    def submit(self, cues):
        if self.failed or not cues:
            return
        try:
            if self.process.poll() is not None:
                raise BrokenPipeError()
            # The pipe bounds queued bytes and provides backpressure. Models
            # stay in a separate process that can be terminated on cancellation.
            with span("queue.submit", cues=len(cues)):
                self.process.stdin.write(json.dumps([asdict(c) for c in cues], ensure_ascii=False)+"\n")
                self.process.stdin.flush()
        except (BrokenPipeError, OSError):
            self.failed = True

    def finish(self, timeout=600):
        try:
            self.process.stdin.close()
        except BrokenPipeError:
            self.failed = True
        with span("queue.drain"):
            try:
                code = self.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                self.failed = True
                self._stop_worker()
                code = self.process.returncode
                self.report(f"Warning: progressive prefetch timed out after {timeout}s; worker terminated")
        self.log.seek(0)
        message = self.log.read()[-2000:].strip()
        if code or self.failed:
            self.report(f"Warning: progressive prefetch failed; continuing with standard cached pipeline: {message}")
        elif message:
            self.report(message)

    def _stop_worker(self):
        if self.process is not None and self.process.poll() is None:
            if os.name == "posix":
                os.killpg(self.process.pid, signal.SIGTERM)
            else:
                self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if os.name == "posix":
                    os.killpg(self.process.pid, signal.SIGKILL)
                else:
                    self.process.kill()
                self.process.wait()

    def close(self):
        self._stop_worker()
        if self.process is not None and not self.process.stdin.closed:
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
        self.log.close()
        self.directory.cleanup()
