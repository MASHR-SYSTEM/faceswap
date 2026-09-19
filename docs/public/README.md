# MASHr FaceSwap

Local camera effects and optional neural face transformation for Windows and Linux.
Original application code is free and open source under Apache-2.0.

**Alpha release candidate:** native packages and platform acceptance are being
validated. This is not a promise of realtime neural performance on every computer.

## Use

Extract the package for your platform and launch `FaceSwap.exe` (Windows) or
`FaceSwap` (Linux). The launcher opens http://127.0.0.1:7865. Choose a camera effect
and press **Start camera**. Camera preview and cartoon require no neural models.
Use **Stop all** to release active camera and audio sessions, then **Exit** in the
launcher to close the server. Closing the browser alone does not stop the app.

Neural face swap requires separately supplied, compatible model files. Those
weights are not included in this repository and are not licensed under the
application licence. See [Setup](SETUP.md) and upstream model terms.

- Website: https://faceswap.mashr.ai/
- Source and support: https://github.com/MASHR-SYSTEM/faceswap
- [Usage](USAGE.md) · [Build from source](BUILD.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

No account or cloud processing is required. Camera/audio processing stays on your
computer. Use your own images or images you have permission to process.
