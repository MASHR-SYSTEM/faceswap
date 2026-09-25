"""Single-owner worker lifecycle. A stuck native call retains ownership until exit."""
from __future__ import annotations
import functools
import threading
from .gpu_jobs import gpu_coordinator


class SessionBusy(RuntimeError):
    pass


def serialized(method):
    @functools.wraps(method)
    def wrapped(self, *args, **kwargs):
        with self._transition_lock:
            return method(self, *args, **kwargs)
    return wrapped


class WorkerLifecycle:
    stop_timeout = 3.0

    def _init_lifecycle(self):
        self._transition_lock = threading.RLock()
        self._generation = 0

    def _require_stopped(self):
        if self._thread is not None and self._thread.is_alive():
            raise SessionBusy('Previous session is still stopping; wait for it to release its resources')

    def _launch(self, argument, name):
        # Called under the transition and state locks; no prior worker can remain alive.
        self._require_stopped()
        try:
            gpu_coordinator.enter_live(self)
        except SessionBusy:
            self._state.running = False
            self._state.phase = 'idle'
            raise
        self._generation += 1
        self._stop_event = threading.Event()
        self._state.generation = self._generation
        self._state.phase = 'starting'
        self._thread = threading.Thread(target=self._worker_entry,
            args=(argument, self._generation), name=name, daemon=True)
        try:
            self._thread.start()
        except Exception:
            gpu_coordinator.leave_live(self)
            self._state.running = False
            self._state.phase = 'error'
            raise

    def _worker_entry(self, argument, generation):
        try:
            self._run(argument)
        except Exception as exc:
            with self._lock:
                self._state.last_error = str(exc)
        finally:
            gpu_coordinator.leave_live(self)
            with self._lock:
                if generation == self._generation:
                    if self._state.last_error == 'Shutdown is still pending; restart is blocked until the worker exits':
                        self._state.last_error = None
                    self._state.running = False
                    self._state.phase = 'error' if self._state.last_error else 'idle'

    def _mark_ready(self):
        with self._lock:
            if not self._stop_event.is_set():
                self._state.phase = 'running'

    def _stop_worker(self):
        with self._lock:
            thread = self._thread
            self._stop_event.set()
            if thread is not None and thread.is_alive():
                self._state.phase = 'stopping'
        if thread and thread.is_alive() and thread is not threading.current_thread():
            thread.join(timeout=self.stop_timeout)
        with self._lock:
            if thread and thread.is_alive():
                self._state.phase = 'stopping'
                self._state.running = True
                self._state.last_error = 'Shutdown is still pending; restart is blocked until the worker exits'
                return False
            self._thread = None
            self._state.running = False
            self._state.phase = 'error' if self._state.last_error else 'idle'
            return True
