# Setup

Release targets: Windows 11 x64 and Ubuntu 24.04 x86-64. ARM and macOS are not
supported by this alpha. CPU builds prioritise compatibility; neural realtime
performance needs a separately verified GPU configuration.

## Install and first launch

Download the Windows alpha installer from https://face.mashr.ai/ and verify its
published SHA-256 checksum. This prerelease is unsigned: Windows will normally show
“Windows protected your PC.” Choose More info and Run anyway only if the checksum
matches and you accept the risk of testing early software. No Python, Node or Git
is needed. The installer creates a
Start menu shortcut and an ordinary Add/Remove Programs entry. The launcher starts a local server and opens
http://127.0.0.1:7865. Use Camera preview or Cartoon in **Setup & help** to test the
standard webcam before installing optional models. IR and virtual cameras must
be selected explicitly. If your camera is busy, close its other owner and retry.

Packaged settings and uploads live under `%LOCALAPPDATA%\MASHr\FaceSwap` on Windows
or `$XDG_DATA_HOME/mashr/faceswap` (normally `~/.local/share/mashr/faceswap`) on Linux.
Replacing the application folder does not replace this data. Source checkouts
keep their existing project-local data; `FACESWAP_DATA_DIR` overrides its location.

## Optional neural models

Use only assets whose terms allow your intended use. InsightFace code is MIT;
its supplied model weights have separate non-commercial research restrictions.
The first-launch setup shows the download size and requires an explicit terms
acknowledgement before downloading directly from the official upstream hosts.
Files resume after network failures and are checked against pinned SHA-256 hashes.

The guided setup installs these files under the models directory shown in **Setup & help**:

```
models/inswapper_128.onnx
models/insightface/models/buffalo_l/*.onnx
models/mediapipe/selfie_segmenter_landscape.tflite
```

Existing valid manually installed files are recognized. The last model is needed
only for background segmentation. A missing/invalid model is reported;
a plain camera image is not a successful neural swap. CPU inference makes no
realtime promise. The Windows package falls back to CPU. NVIDIA acceleration
requires compatible NVIDIA drivers plus CUDA 13 and cuDNN 9 installed separately.

## Optional virtual camera

On Linux install `v4l2loopback-dkms` and create a device, for example:

```
sudo modprobe v4l2loopback devices=1 video_nr=42 card_label=FaceSwap exclusive_caps=1
```

Enable Virtual camera and start/restart the camera; select FaceSwap in the
receiving application. Kernel/module compatibility is a system prerequisite.

On Windows install UnityCapture separately following its upstream instructions:
https://github.com/schellingb/UnityCapture#installation . The filter is not bundled.
Enable Virtual camera in FaceSwap and select **Unity Video Capture** in the
receiving application. Waiting for receiver means no application is consuming it
yet. OBS can receive this output but does not have to produce it.

## Optional voice

Windows supports DSP voice. Install VB-CABLE from https://vb-audio.com/Cable/
using the vendor's instructions, including any required restart. Select your
physical microphone as Input and **CABLE Input** as Virtual voice output in
FaceSwap. Select **CABLE Output** as microphone in the receiving application.
VB-CABLE is separately licensed and is not part of FaceSwap's FOSS distribution.
Headphone monitoring has its own output control; avoid routing it back to input.

Linux voice uses PipeWire/PulseAudio and requires PortAudio plus `pactl`/`pacat`
utilities. The call application selects FaceSwap_Voice_Mic.

Install optional system devices yourself; FaceSwap does not silently install
or elevate drivers. Missing optional devices do not prevent camera preview.
