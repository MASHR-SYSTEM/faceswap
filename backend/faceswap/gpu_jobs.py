"""Conservative admission: live sessions share GPU; offline rendering is exclusive."""
import threading
from contextlib import contextmanager


class JobCancelled(RuntimeError):
    pass


class GpuCoordinator:
    def __init__(self):
        self.condition = threading.Condition()
        self.live = set()
        self.offline = False

    def enter_live(self, owner):
        from .lifecycle import SessionBusy
        with self.condition:
            if self.offline:
                raise SessionBusy('An offline render owns the GPU; cancel it or wait before starting live output')
            self.live.add(owner)

    def leave_live(self, owner):
        with self.condition:
            self.live.discard(owner)
            self.condition.notify_all()

    @contextmanager
    def render(self, cancelled):
        with self.condition:
            while self.live or self.offline:
                if cancelled():
                    raise JobCancelled('Render cancelled while waiting for GPU')
                self.condition.wait(.1)
            if cancelled():
                raise JobCancelled('Render cancelled')
            self.offline = True
        try:
            yield
        finally:
            with self.condition:
                self.offline = False
                self.condition.notify_all()


gpu_coordinator = GpuCoordinator()
