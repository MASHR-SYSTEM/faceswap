# Candidate tester checklist

Windows 11 x64 physical-device acceptance is pending. Automated Windows package
build and startup checks do not establish camera or receiving-app compatibility.
Please record the archive checksum, Windows build, camera model, output driver
version, receiving app/version and whether CPU or CUDA runtime is installed.

1. Extract the archive to a fresh directory and open FaceSwap.exe. Confirm the
   local browser opens; no Python or Node installation should be necessary.
2. Start camera preview without models. Automatic selection should use the
   standard RGB camera, excluding IR and virtual devices. Repeat start/stop ten
   times, including a rapid stop during startup. Confirm the camera is released.
3. Try cartoon/privacy effects, choose/change/remove a background, and cancel the
   image picker. Invalid uploads should preserve the current background.
4. With separately licensed models supplied, verify the neural effect and blend
   slider. Record frame rate and errors; model availability is a separate check.
5. Install UnityCapture separately following SETUP.md. Enable virtual video,
   select it in a receiving app, and verify colour, dimensions and motion for
   30 minutes. Disconnect/reconnect the receiver and confirm recovery.
6. Optional voice: install VB-CABLE separately, select a physical microphone and
   CABLE Input output. Confirm DSP audio reaches the receiving app through CABLE
   Output. Stop all must release video and audio; no output should continue.
7. Change camera or voice settings while running: confirm restart is required
   and the current session is unchanged until restart. Test backend restart and
   disconnected status, small windows and keyboard navigation.
8. Exit using the launcher. Confirm the owned backend stops and camera/audio
   devices are available to another application. Relaunch twice to verify the
   existing instance is reused and closing a second launcher does not stop it.

For each item record PASS / FAIL / NOT TESTED, observed results and reproducible
steps for failures. Avoid attaching personal face images or model weights.
A candidate is not a verified public release until required tests pass.
