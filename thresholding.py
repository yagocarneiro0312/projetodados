"""Segmentação fixa ou local para faces claras e pontos escuros."""

import cv2


def threshold_image(gray, mode, cutoff, block_size, constant, inverted=False, mask=None):
    threshold_type = cv2.THRESH_BINARY_INV if inverted else cv2.THRESH_BINARY
    if mode == "fixed":
        return cv2.threshold(gray, cutoff, 255, threshold_type)[1]
    if mode == "otsu":
        # A mesa e os pixels excluídos pela máscara não calibram os pontos.
        samples = gray[mask > 0] if mask is not None else gray
        if not samples.size:
            return gray * 0
        cutoff = cv2.threshold(samples, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
        return cv2.threshold(gray, cutoff, 255, threshold_type)[1]
    method = {
        "adaptive_gaussian": cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        "adaptive_mean": cv2.ADAPTIVE_THRESH_MEAN_C,
    }[mode]
    return cv2.adaptiveThreshold(gray, 255, method, threshold_type, block_size, constant)
