# Face-swap backends

The camera/session and virtual-camera pipeline use the `FaceSwapBackend` protocol
in `backend/faceswap/swap_backends.py`. InSwapper is the only built-in adapter.
This change does not add HyperSwap weights or claim compatibility with arbitrary
ONNX files.

To add an engine, implement the protocol in a new module and register a
`BackendSpec` at application startup, before creating sessions or validating
configuration. For built-in adapters, add the registration beside InSwapper's in
`swap_backends.py`. Use a lazy factory to avoid loading optional dependencies at
startup:

```python
def create_example(models_dir, cache_dir):
    from .example_backend import ExampleBackend
    return ExampleBackend(models_dir, cache_dir)

register_backend(BackendSpec(
    "example", "Example engine", "example.onnx", create_example,
))
```

The adapter owns face detection, source identity encoding, inference, alignment,
blending, and model-specific effect controls. It accepts BGR uint8 frames and
returns the same dimensions and format. Face/mouth geometry is in frame pixels;
return `None` when unavailable. Report timing, provider, readiness, and errors
through the protocol properties. Missing models/inference failures should set
the error and return the original frame. `close()` must release model resources
and allow a later `configure()` call. Factories should be lightweight.

The shared processor still applies background replacement and lip animation.
Preview, capture, and virtual-camera output require no adapter-specific changes.
Each backend gets a separate cache directory. Switching engines closes the old
adapter and constructs a new one on the processing thread; changes to the source
image or other settings are passed to the existing adapter's `configure()`.

`EffectConfig.swap_backend` selects a registered ID and defaults to `inswapper`
for existing saved settings/API clients. Unknown IDs are rejected by API
validation. `/api/capabilities` lists registered engines in `swap_backends`; the
Performance panel builds its selector from that list. Registration does not mean
that model files or runtime dependencies are installed.

The UI clears a custom model path when changing engines. API callers should also
set `model_path` to null when switching to use the new engine's default filename,
or provide a model compatible with that engine. Provider and precision requests
are passed to the adapter, which must implement or explicitly report unsupported
settings. The existing `onnx_faceswap` effect-mode name is retained for API
compatibility; adapters need not use ONNX internally.
