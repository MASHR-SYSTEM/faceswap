# MASHr FaceSwap

Local camera effects and optional neural face transformation for Windows and Linux.
Original application code is free and open source under Apache-2.0.

**Alpha release candidate:** native packages and platform acceptance are being
validated. This is not a promise of realtime neural performance on every computer.

## Use

On Windows, download the alpha installer from the website, verify its published
SHA-256 checksum, and run it. This prerelease is unsigned, so Windows will normally
show a SmartScreen warning. Then launch
**MASHr FaceSwap** from the Start menu. The launcher opens http://127.0.0.1:7865. Choose a camera effect
and press **Start camera**. Camera preview and cartoon require no neural models.
Use **Stop all** to release active camera and audio sessions, then **Exit** in the
launcher to close the server. Closing the browser alone does not stop the app.

On first use, the app offers to download the current optional models after you
acknowledge their separate non-commercial research terms. Downloads are resumed,
checksum-verified, and stored locally; camera preview works if you skip them.

- Website and Windows download: https://face.mashr.ai/
- Source and support: https://github.com/MASHR-SYSTEM/faceswap
- [Usage](USAGE.md) · [Build from source](BUILD.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

No account or cloud processing is required. Camera/audio processing stays on your
computer. Use your own images or images you have permission to process.
