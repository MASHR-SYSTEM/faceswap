"""TTS output with a PulseAudio server playback clock (also works on PipeWire)."""
from __future__ import annotations
import ctypes as C
import ctypes.util
import queue
import threading
import time


class _SampleSpec(C.Structure):
    _fields_ = [('format', C.c_int), ('rate', C.c_uint32), ('channels', C.c_uint8)]


class _BufferAttr(C.Structure):
    _fields_ = [(name, C.c_uint32) for name in ('maxlength', 'tlength', 'prebuf', 'minreq', 'fragsize')]


class PulseSink:
    def __init__(self, sample_rate, channels, sink_name):
        library = ctypes.util.find_library('pulse-simple')
        if not library:
            raise RuntimeError('libpulse-simple is required for clocked speech playback')
        self.lib = C.CDLL(library)
        self.lib.pa_simple_new.restype = C.c_void_p
        self.lib.pa_simple_new.argtypes = [C.c_char_p, C.c_char_p, C.c_int, C.c_char_p,
            C.c_char_p, C.POINTER(_SampleSpec), C.c_void_p, C.c_void_p, C.POINTER(C.c_int)]
        self.lib.pa_simple_write.argtypes = [C.c_void_p, C.c_void_p, C.c_size_t, C.POINTER(C.c_int)]
        self.lib.pa_simple_get_latency.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self.lib.pa_simple_get_latency.restype = C.c_uint64
        self.lib.pa_simple_free.argtypes = [C.c_void_p]
        self.lib.pa_simple_flush.argtypes = [C.c_void_p, C.POINTER(C.c_int)]
        self.error = C.c_int()
        self.handle = self.lib.pa_simple_new(None, b'FaceSwap speech', 1,
            sink_name.encode() if sink_name else None, b'TTS playback',
            C.byref(_SampleSpec(5, sample_rate, channels)), None,
            C.byref(_BufferAttr(0xffffffff, int(sample_rate * channels * 4 * .04),
                               0xffffffff, 0xffffffff, 0xffffffff)), C.byref(self.error))
        if not self.handle:
            raise RuntimeError(f'PulseAudio output unavailable (error {self.error.value})')

    def write(self, payload):
        buffer = C.create_string_buffer(payload)
        if self.lib.pa_simple_write(self.handle, buffer, len(payload), C.byref(self.error)) < 0:
            raise RuntimeError(f'PulseAudio write failed ({self.error.value})')

    def latency(self):
        micros = self.lib.pa_simple_get_latency(self.handle, C.byref(self.error))
        if micros == 2**64 - 1:
            raise RuntimeError(f'PulseAudio latency query failed ({self.error.value})')
        return micros / 1_000_000

    def close(self):
        if self.handle:
            self.lib.pa_simple_flush(self.handle, C.byref(self.error))
            self.lib.pa_simple_free(self.handle)
            self.handle = None


class PlaybackWriter:
    """Bounded writer; clock only advances through samples submitted to the server.

    Generation stalls cannot move the clock past the last submitted sample. Server
    latency includes its queued samples, unlike a timestamp at queue insertion.
    """
    def __init__(self, *, sample_rate, channels=1, sink_name=None, queue_blocks=24,
                 sink_factory=PulseSink, clock=time.monotonic):
        self.rate, self.channels = sample_rate, channels
        self.sink_name, self._factory, self._clock = sink_name, sink_factory, clock
        self._queue = queue.Queue(maxsize=queue_blocks)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread = None
        self._submitted = 0.0
        self._played = 0.0
        self._observed = 0.0
        self.failed, self.last_error = False, None
        self._sink = None

    def start(self):
        self._sink = self._factory(self.rate, self.channels, self.sink_name)
        self._thread = threading.Thread(target=self._run, name='faceswap-playback', daemon=True)
        self._thread.start()

    def write(self, payload, timeout=0.0):
        if self.failed or self._stop.is_set():
            return False
        try:
            self._queue.put(payload, timeout=timeout)
            return True
        except queue.Full:
            return False

    def position_seconds(self):
        with self._lock:
            if not self._submitted:
                return 0.0
            return max(0.0, min(self._submitted, self._played + self._clock() - self._observed))

    def _run(self):
        try:
            while not self._stop.is_set():
                try:
                    payload = self._queue.get(timeout=.05)
                except queue.Empty:
                    continue
                self._sink.write(payload)
                latency = self._sink.latency()
                with self._lock:
                    self._submitted += len(payload) / (4 * self.channels * self.rate)
                    self._played = self._submitted - latency
                    self._observed = self._clock()
        except Exception as exc:
            self.failed, self.last_error = True, str(exc)
        finally:
            self._sink.close()

    def close(self):
        self._stop.set()
        if self._thread:
            # The session's lifecycle timeout retains ownership if native output is stuck.
            self._thread.join()
