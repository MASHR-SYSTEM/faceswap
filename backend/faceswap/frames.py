"""Latest-only frame handoff with explicit ownership across worker threads."""
import threading
import time


class LatestCapture:
    def __init__(self, capture, first_frame=None):
        self.capture = capture
        self.condition = threading.Condition()
        self.stop_event = threading.Event()
        self.frame = first_frame.copy(order='C') if first_frame is not None else None
        self.sequence = 1 if first_frame is not None else 0
        self.timestamp = time.monotonic() if first_frame is not None else 0.0
        self.error = None
        self.thread = threading.Thread(target=self._read, name='faceswap-capture', daemon=True)

    def start(self):
        self.thread.start()
        return self

    def _read(self):
        try:
            while not self.stop_event.is_set():
                ok, frame = self.capture.read()
                if not ok or frame is None:
                    raise RuntimeError('Camera frame read failed')
                # Some Windows camera drivers recycle the memory behind the
                # ndarray returned by VideoCapture.read().  The processing
                # thread must never observe that buffer while the driver is
                # writing the next frame into it.
                frame = frame.copy(order='C')
                with self.condition:
                    self.frame = frame
                    self.sequence += 1
                    self.timestamp = time.monotonic()
                    self.condition.notify_all()
        except Exception as exc:
            with self.condition:
                self.error = str(exc)
                self.condition.notify_all()
        finally:
            self.capture.release()

    def next(self, after, timeout=.1):
        with self.condition:
            self.condition.wait_for(lambda: self.sequence > after or self.error or self.stop_event.is_set(), timeout)
            if self.error:
                raise RuntimeError(self.error)
            if self.sequence <= after:
                return None
            # Keep ownership unambiguous even for capture implementations that
            # return a view backed by driver-managed memory.
            return self.sequence, self.timestamp, self.frame.copy(order='C')

    def close(self):
        self.stop_event.set()
        with self.condition:
            self.condition.notify_all()
        # A stuck camera retains the parent session's ownership via its join guard.
        self.thread.join()


class LatestOutput:
    def __init__(self, camera):
        self.camera = camera
        self.condition = threading.Condition()
        self.stopped = False
        self.frame = None
        self.error = None
        self.thread = threading.Thread(target=self._send, name='faceswap-output', daemon=True)
        self.thread.start()

    def publish(self, rgb):
        with self.condition:
            if self.error:
                raise RuntimeError(self.error)
            self.frame = rgb
            self.condition.notify_all()

    def _send(self):
        try:
            while True:
                with self.condition:
                    self.condition.wait_for(lambda: self.frame is not None or self.stopped)
                    if self.stopped:
                        return
                    frame = self.frame
                self.camera.send(frame)
                self.camera.sleep_until_next_frame()
        except Exception as exc:
            self.error = str(exc)
        finally:
            self.camera.close()

    def close(self):
        with self.condition:
            self.stopped = True
            self.condition.notify_all()
        self.thread.join()
