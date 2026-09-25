"""Aligned InSwapper inference with a cached source projection and local paste-back."""
import time
import cv2
import numpy as np


class _CudaIoRunner:
    """Keep fixed-shape swap tensors on CUDA between camera frames."""

    def __init__(self, model, blob, latent):
        import onnxruntime as ort

        self.session = model.session
        self.image = ort.OrtValue.ortvalue_from_numpy(blob, "cuda", 0)
        self.latent = ort.OrtValue.ortvalue_from_numpy(latent, "cuda", 0)
        self.binding = self.session.io_binding()
        self.binding.bind_ortvalue_input(model.input_names[0], self.image)
        self.binding.bind_ortvalue_input(model.input_names[1], self.latent)
        # Let ORT allocate the output on CUDA.  Only the small final 128x128
        # result crosses back to the CPU for paste-back.
        self.binding.bind_output(model.output_names[0], "cuda", 0)

    def run(self, blob):
        self.image.update_inplace(blob)
        self.session.run_with_iobinding(self.binding)
        return self.binding.copy_outputs_to_cpu()[0]


def _run_swap_model(model, blob, latent):
    providers = model.session.get_providers()
    if providers and providers[0] == "CUDAExecutionProvider":
        runner = getattr(model, "_mashr_cuda_io_runner", None)
        if runner is None:
            try:
                runner = _CudaIoRunner(model, blob, latent)
            except Exception:
                # Older ORT/CUDA combinations may not expose device OrtValues.
                # Remember the failure and retain the compatible run path.
                runner = False
            model._mashr_cuda_io_runner = runner
        if runner:
            try:
                return runner.run(blob)
            except Exception:
                # I/O binding is an optimization, never a reason to lose video.
                model._mashr_cuda_io_runner = False
    return model.session.run(model.output_names,
        {model.input_names[0]: blob, model.input_names[1]: latent})[0]


def swap_frame(model, frame, face, latent, timings=None):
    from insightface.utils.face_align import norm_crop2
    crop, transform = norm_crop2(frame, face.kps, model.input_size[0])
    blob = cv2.dnn.blobFromImage(crop, 1.0 / model.input_std, model.input_size,
                               (model.input_mean,) * 3, swapRB=True)
    started = time.perf_counter()
    prediction = _run_swap_model(model, blob, latent)
    if timings is not None:
        timings.inference_ms = (time.perf_counter() - started) * 1000
    if not np.isfinite(prediction).all():
        raise RuntimeError("Neural runtime returned non-finite pixels; select FP32 or a validated precision preset")
    generated = np.clip(prediction[0].transpose(1, 2, 0) * 255, 0, 255).astype(np.uint8)[:, :, ::-1]
    inverse = cv2.invertAffineTransform(transform)
    height, width = frame.shape[:2]
    # Determine support before warping, avoiding three full-resolution warp buffers.
    corners = cv2.transform(np.array([[[0., 0.], [crop.shape[1], 0.],
        [crop.shape[1], crop.shape[0]], [0., crop.shape[0]]]], np.float32), inverse)[0]
    halo = max(32, int(np.max(np.ptp(corners, axis=0)) / 10) + 8)
    x, y = np.maximum(0, np.floor(corners.min(axis=0)).astype(int) - halo)
    right, bottom = np.minimum([width, height], np.ceil(corners.max(axis=0)).astype(int) + halo)
    if right <= x or bottom <= y:
        return frame
    local = inverse.copy()
    local[:, 2] -= [x, y]
    size = (int(right-x), int(bottom-y))
    generated = cv2.warpAffine(generated, local, size)
    mask = cv2.warpAffine(np.full(crop.shape[:2], 255, np.float32), local, size)
    mask[mask > 20] = 255
    ys, xs = np.where(mask == 255)
    if not len(xs):
        return frame
    extent = int(np.sqrt((ys.max()-ys.min()) * (xs.max()-xs.min())))
    erosion = max(extent // 10, 10)
    mask = cv2.erode(mask, np.ones((erosion, erosion), np.uint8))
    radius = max(extent // 20, 5)
    mask = cv2.GaussianBlur(mask, (2*radius+1, 2*radius+1), 0)[:, :, None] / 255
    output = frame.copy()
    output[y:bottom, x:right] = (mask * generated + (1-mask) * frame[y:bottom, x:right]).astype(np.uint8)
    return output
