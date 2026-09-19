# Third-party components

The application licence does not replace dependency licences. Release archives
must include the notices collected from installed distributions by the package
build. Model weights, NVIDIA drivers and virtual audio/camera system devices
are optional, separately licensed components.

- React / Vite / Lucide: MIT / ISC notices distributed with their packages.
- FastAPI / Pydantic / Starlette / ONNX Runtime / InsightFace code: MIT.
- NumPy / SciPy: BSD licences and included native library notices.
- OpenCV: Apache-2.0, with separate bundled codec/library notices.
- Pillow: HPND and bundled library notices.
- UnityCapture protocol interoperability: Bernhard Schelling and MHD Yamen Saraiji;
  upstream filter MIT / plugin zlib. The filter itself is not redistributed.
- PyInstaller build tooling: GPL with its distribution exception; build-time only.

InsightFace neural weights are not covered by its MIT code licence and are not
automatically downloaded. Consult the upstream model terms before use.

Release gate: review the generated dependency inventory and collected notices for
each platform build before distributing it. This list is not a blanket licence
clearance for optional native libraries.
