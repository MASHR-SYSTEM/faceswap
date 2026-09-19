from pathlib import Path
import platform

if platform.system() == "Linux":
    import fcntl
import os
import re
import struct

from .schemas import DeviceInfo


SYS_VIDEO = Path("/sys/class/video4linux")
DEV_VIDEO = Path("/dev")
# Linux videodev2.h: _IOR('V', 0, struct v4l2_capability), 104 bytes.
VIDIOC_QUERYCAP = 0x80685600
V4L2_CAP_DEVICE_CAPS = 0x80000000
V4L2_CAP_VIDEO_CAPTURE = 0x00000001
V4L2_CAP_VIDEO_CAPTURE_MPLANE = 0x00001000


def _query_capabilities(path: Path) -> tuple[str, int]:
    # Query metadata only. Never allocate buffers or start/stop a camera stream.
    fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK)
    try:
        data = bytearray(104)
        fcntl.ioctl(fd, VIDIOC_QUERYCAP, data)
        driver, _card, _bus, _version, caps, device_caps, *_ = struct.unpack("=16s32s32s6I", data)
        return driver.split(b"\0")[0].decode(errors="replace"), device_caps if caps & V4L2_CAP_DEVICE_CAPS else caps
    finally:
        os.close(fd)


def list_camera_devices(max_devices: int | None = None) -> list[DeviceInfo]:
    if platform.system() == "Windows":
        return _windows_cameras()
    devices: list[DeviceInfo] = []
    for node in SYS_VIDEO.glob("video*"):
        if not node.name[5:].isdigit():
            continue
        index = int(node.name[5:])
        if max_devices is not None and index >= max_devices:
            continue
        try:
            name = (node / "name").read_text().strip()
        except OSError:
            continue
        virtual = "/virtual/" in str(node.resolve())
        try:
            driver, caps = _query_capabilities(DEV_VIDEO / node.name)
            if not caps & (V4L2_CAP_VIDEO_CAPTURE | V4L2_CAP_VIDEO_CAPTURE_MPLANE):
                continue  # Metadata nodes are not image sources.
            virtual = virtual or driver == "v4l2loopback"
        except OSError:
            # Busy/permission-limited devices must remain discoverable. For UVC
            # fallback, index 0 is the image node; secondary metadata is excluded.
            try:
                if (node / "index").read_text().strip() != "0":
                    continue
            except OSError:
                continue
        infrared = bool(re.search(r"\b(ir|infrared)\b|infra-red", name, re.IGNORECASE))
        kind = "virtual" if virtual else "infrared" if infrared else "standard"
        prefix = {"standard": "Standard", "infrared": "Infrared", "virtual": "Virtual"}[kind]
        devices.append(DeviceInfo(index=index, kind=kind, device_id=_linux_device_id(node),
            label=f"{prefix}: {name} (/dev/video{index})"))
    devices.sort(key=lambda device: ({"standard": 0, "infrared": 1, "virtual": 2}[device.kind], device.index))
    if devices and devices[0].kind == "standard":
        devices[0].is_default = True
    return devices


def list_v4l2loopback_devices() -> list[str]:
    devices: list[str] = []
    for path in sorted(Path("/sys/devices/virtual/video4linux").glob("video*")):
        name_file = path / "name"
        try:
            name = name_file.read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if "loopback" in name.lower() or "faceswap" in name.lower() or "dummy" in name.lower():
            devices.append(f"/dev/{path.name} ({name})")
    return devices


def _linux_device_id(node: Path) -> str:
    for link in sorted(Path('/dev/v4l/by-id').glob('*')):
        if link.resolve() == (DEV_VIDEO / node.name).resolve():
            return str(link)
    return str(DEV_VIDEO / node.name)


def _windows_cameras() -> list[DeviceInfo]:
    from cv2_enumerate_cameras import enumerate_cameras
    import cv2
    devices = []
    for camera in enumerate_cameras(cv2.CAP_DSHOW):
        name = camera.name
        virtual = bool(re.search(r'virtual|unity|obs|manycam', name, re.I))
        infrared = bool(re.search(r'\b(ir|infrared)\b|infra-red', name, re.I))
        kind = 'virtual' if virtual else 'infrared' if infrared else 'standard'
        devices.append(DeviceInfo(index=camera.index, device_id=camera.path or None,
                                  label=f'{kind.title()}: {name}', kind=kind))
    devices.sort(key=lambda d: ({'standard': 0, 'infrared': 1, 'virtual': 2}[d.kind], d.index))
    if devices and devices[0].kind == 'standard':
        devices[0].is_default = True
    return devices


def capture_backend():
    import cv2
    return cv2.CAP_DSHOW if platform.system() == 'Windows' else cv2.CAP_V4L2
