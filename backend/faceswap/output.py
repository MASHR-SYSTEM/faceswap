"""Direct virtual outputs. No bundled virtual-camera driver or GPL wrapper.

Linux uses the documented V4L2 userspace ABI. Windows interoperates with the
UnityCapture shared-memory protocol (upstream: schellingb/UnityCapture,
Bernhard Schelling / MHD Yamen Saraiji; see THIRD_PARTY_NOTICES.md).
"""
import os
import platform
import struct
import time

import numpy as np


class PacedOutput:
    def __init__(self, width, height, fps):
        self.width, self.height = width, height
        self.period = 1 / max(1, fps)
        self.deadline = time.monotonic()
        self.state = 'ready'

    def sleep_until_next_frame(self):
        self.deadline = max(self.deadline + self.period, time.monotonic())
        time.sleep(min(self.period, max(0, self.deadline - time.monotonic())))

    def validate(self, frame):
        if frame.shape != (self.height, self.width, 3) or frame.dtype != np.uint8:
            raise ValueError('Virtual output requires an RGB uint8 frame of the configured dimensions')


class V4L2Output(PacedOutput):
    def __init__(self, width, height, fps, device=None):
        super().__init__(width, height, fps)
        import fcntl
        from .camera import list_v4l2loopback_devices, _query_capabilities
        if device is None:
            devices = list_v4l2loopback_devices()
            if not devices:
                raise RuntimeError('Install and load v4l2loopback to use virtual camera output. See Setup.')
            device = devices[0].split(' ')[0]
        driver, _ = _query_capabilities(__import__('pathlib').Path(device))
        if driver != 'v4l2loopback':
            raise RuntimeError('The output device must be a v4l2loopback camera')
        self.fd = os.open(device, os.O_WRONLY | os.O_NONBLOCK)
        try:
            # v4l2_format: type + alignment + 200-byte union on supported x86-64.
            fmt = bytearray(208)
            struct.pack_into('=I', fmt, 0, 2)  # VIDEO_OUTPUT
            rgb24 = int.from_bytes(b'RGB3', 'little')
            struct.pack_into('=8I', fmt, 8, width, height, rgb24, 1, width * 3, width * height * 3, 8, 0)
            fcntl.ioctl(self.fd, 0xC0D05605, fmt)  # VIDIOC_S_FMT
            actual_w, actual_h, fourcc, _, stride, size = struct.unpack_from('=6I', fmt, 8)
            if (actual_w, actual_h, fourcc) != (width, height, rgb24) or stride < width * 3 or size < stride * height:
                raise RuntimeError('Virtual camera rejected the requested RGB frame format')
            self.stride = stride
            self.size = size
            self.device = device
        except BaseException:
            self.close()
            raise

    def send(self, frame):
        self.validate(frame)
        payload = np.zeros(self.size, np.uint8)
        for row in range(self.height):
            payload[row * self.stride:row * self.stride + self.width * 3] = frame[row].reshape(-1)
        try:
            written = os.write(self.fd, payload.tobytes())
        except BlockingIOError:
            self.state = 'waiting_for_receiver'
            return
        if written != self.size:
            raise RuntimeError('Virtual camera accepted an incomplete frame')
        self.state = 'ready'

    def close(self):
        if getattr(self, 'fd', None) is not None:
            os.close(self.fd)
            self.fd = None


class UnityOutput(PacedOutput):
    """Windows-only ctypes sender; filter installation stays user-controlled."""
    def __init__(self, width, height, fps):
        super().__init__(width, height, fps)
        import ctypes as c
        from ctypes import wintypes as w
        self.c = c
        self.k = c.WinDLL('kernel32', use_last_error=True)
        signatures = {
            'OpenMutexW': ([w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
            'CreateMutexW': ([c.c_void_p, w.BOOL, w.LPCWSTR], w.HANDLE),
            'OpenEventW': ([w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
            'CreateEventW': ([c.c_void_p, w.BOOL, w.BOOL, w.LPCWSTR], w.HANDLE),
            'OpenFileMappingW': ([w.DWORD, w.BOOL, w.LPCWSTR], w.HANDLE),
            'MapViewOfFile': ([w.HANDLE, w.DWORD, w.DWORD, w.DWORD, c.c_size_t], c.c_void_p),
            'WaitForSingleObject': ([w.HANDLE, w.DWORD], w.DWORD),
            'ReleaseMutex': ([w.HANDLE], w.BOOL), 'SetEvent': ([w.HANDLE], w.BOOL),
            'CloseHandle': ([w.HANDLE], w.BOOL), 'UnmapViewOfFile': ([c.c_void_p], w.BOOL),
        }
        for name, (args, result) in signatures.items():
            function = getattr(self.k, name)
            function.argtypes, function.restype = args, result
        self.handles = []
        self.mapping = None
        self.publisher = self.k.CreateMutexW(None, False, 'Local\\MASHrFaceSwapUnityPublisher')
        if not self.publisher:
            raise OSError(c.get_last_error(), 'Cannot create output ownership mutex')
        self.owns_publisher = False
        self.state = 'waiting_for_receiver'
        self.next_connect = 0.0
        self.last_receiver = time.monotonic()

    def _disconnect(self):
        if self.mapping:
            self.k.UnmapViewOfFile(self.mapping)
            self.mapping = None
        for handle in self.handles:
            self.k.CloseHandle(handle)
        self.handles.clear()

    def _connect(self):
        if time.monotonic() < self.next_connect:
            return False
        self.next_connect = time.monotonic() + .5
        self.mutex = self.k.OpenMutexW(0x00100001, False, 'UnityCapture_Mutx')
        if not self.mutex:
            return False
        self.handles.append(self.mutex)
        self.want = self.k.CreateEventW(None, False, False, 'UnityCapture_Want')
        self.sent = self.k.OpenEventW(2, False, 'UnityCapture_Sent')
        shared = self.k.OpenFileMappingW(2, False, 'UnityCapture_Data')
        for handle in (self.want, self.sent, shared):
            if handle:
                self.handles.append(handle)
        if not all((self.want, self.sent, shared)):
            self._disconnect()
            return False
        self.mapping = self.k.MapViewOfFile(shared, 2, 0, 0, 36 + self.width * self.height * 4)
        if not self.mapping:
            self._disconnect()
            return False
        return True

    def send(self, frame):
        self.validate(frame)
        if not self.owns_publisher:
            result = self.k.WaitForSingleObject(self.publisher, 0)
            if result not in (0, 0x80):
                raise RuntimeError('Another FaceSwap instance owns Unity virtual camera output')
            self.owns_publisher = True
        if not self.mapping and not self._connect():
            self.state = 'waiting_for_receiver'
            return
        result = self.k.WaitForSingleObject(self.mutex, 20)
        if result not in (0, 0x80):
            self.state = 'waiting_for_receiver'
            return
        try:
            rgba = np.empty((self.height, self.width, 4), np.uint8)
            rgba[:, :, :3], rgba[:, :, 3] = frame, 255
            size = rgba.nbytes
            capacity = self.c.c_uint32.from_address(self.mapping).value
            if capacity < size or capacity > 3840 * 2160 * 8:
                raise RuntimeError('Virtual camera shared buffer has invalid dimensions')
            # Preserve receiver capacity; publish width/height/stride/format/resize/mirror/timeout.
            header = struct.pack('=8i', self.width, self.height, self.width, 0, 0, 0, 1000, 0)
            self.c.memmove(self.mapping + 4, header[:28], 28)
            self.c.memmove(self.mapping + 32, rgba.ctypes.data, size)
        finally:
            self.k.ReleaseMutex(self.mutex)
        self.k.SetEvent(self.sent)
        receiving = self.k.WaitForSingleObject(self.want, 0) == 0
        self.state = 'ready' if receiving else 'waiting_for_receiver'
        if receiving:
            self.last_receiver = time.monotonic()
        if not receiving and time.monotonic() - self.last_receiver > 2:
            # Receiver owns transport. Drop our references so a later receiver
            # can create a fresh mapping after a crash or application restart.
            self._disconnect()

    def close(self):
        self._disconnect()
        if getattr(self, 'publisher', None):
            if self.owns_publisher:
                self.k.ReleaseMutex(self.publisher)
            self.k.CloseHandle(self.publisher)
            self.publisher = None


def open_output(width, height, fps, device=None):
    if platform.system() == 'Windows':
        return UnityOutput(width, height, fps)
    if platform.system() == 'Linux':
        return V4L2Output(width, height, fps, device)
    raise RuntimeError('Virtual output is supported on Windows and Linux only')
