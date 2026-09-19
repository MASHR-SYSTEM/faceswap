# Setup

Release targets: Windows 11 x64 and Ubuntu 24.04 x86-64. ARM and macOS are not
supported by this alpha. CPU builds prioritise compatibility; neural realtime
performance needs a separately verified GPU configuration.

## First launch

Extract the entire archive before running the launcher. No Python, Node or Git
is needed for a packaged release. The launcher starts a local server and opens
http://127.0.0.1:7865. Use Camera preview or Cartoon in **Setup & help** to test the
standard webcam before installing optional models. IR and virtual cameras must
be selected explicitly. If your camera is busy, close its other owner and retry.

Packaged settings and uploads live under `%LOCALAPPDATA%\MASHr\FaceSwap` on Windows
or `$XDG_DATA_HOME/mashr/faceswap` (normally `~/.local/share/mashr/faceswap`) on Linux.
Replacing the application folder does not replace this data. Source checkouts
keep their existing project-local data; `FACESWAP_DATA_DIR` overrides its location.

## Optional neural models

Use only assets whose terms allow your intended use. InsightFace code is MIT;
its supplied model weights have separate restrictions. FaceSwap neither supplies
those weights nor automatically downloads them. Consult
https://github.com/deepinsight/insightface#license before obtaining models.

Place compatible files under the models directory shown in **Setup & help**:

```
models/inswapper_128.onnx
models/insightface/models/buffalo_l/*.onnx
models/mediapipe/selfie_segmenter_landscape.tflite
```

The last model is needed only for background segmentation and has its own source
and terms. Use an optional neural-enabled package/runtime, upload a target image,
select Neural face swap and refresh devices. A missing/invalid model is reported;
a plain camera image is not a successful neural swap. CPU inference makes no
realtime promise. CUDA requires compatible NVIDIA drivers/runtime libraries.

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
