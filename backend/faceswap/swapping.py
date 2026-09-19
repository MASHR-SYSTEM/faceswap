"""Aligned InSwapper inference with a cached source projection and local paste-back."""
import time
import cv2
import numpy as np


def swap_frame(model, frame, face, latent, timings=None):
    from insightface.utils.face_align import norm_crop2
    crop, transform = norm_crop2(frame, face.kps, model.input_size[0])
    blob = cv2.dnn.blobFromImage(crop, 1.0 / model.input_std, model.input_size,
                               (model.input_mean,) * 3, swapRB=True)
    started = time.perf_counter()
    prediction = model.session.run(model.output_names,
        {model.input_names[0]: blob, model.input_names[1]: latent})[0]
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
