"""Cenas determinísticas de dados; nenhuma anotação entra no detector."""

import cv2
import numpy as np
from functools import lru_cache


PATTERNS = {
    0: [],
    1: [(0.5, 0.5)],
    2: [(0.25, 0.25), (0.75, 0.75)],
    3: [(0.25, 0.25), (0.5, 0.5), (0.75, 0.75)],
    4: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.75), (0.75, 0.75)],
    5: [(0.25, 0.25), (0.75, 0.25), (0.5, 0.5), (0.25, 0.75), (0.75, 0.75)],
    6: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.75, 0.5),
        (0.25, 0.75), (0.75, 0.75)],
    7: [(0.25, 0.25), (0.75, 0.25), (0.25, 0.5), (0.5, 0.5),
        (0.75, 0.5), (0.25, 0.75), (0.75, 0.75)],
}


@lru_cache(maxsize=8)
def _table(background):
    yy, xx = np.indices((562, 1000), dtype=np.float32)
    table = background + 9 * (xx / 1000 - .5) + 7 * (yy / 562 - .5)
    return np.repeat(table[:, :, None], 3, axis=2)


def render_scene(value=6, angle=0, center=(500, 280), side=42, background=155,
                 face=228, pip_radius=.082, perspective=0, blur=.45, noise=0, seed=17):
    """Renderiza em 4×, com cantos arredondados, projeção e luz desigual.

    A saída BGR tem 1000×562. O fundo lembra uma mesa clara; a inclinação
    comprime um eixo e desloca dois cantos sem usar o valor na geometria.
    """
    image = _table(background).copy()
    size = 256
    mask = np.zeros((size, size), np.uint8)
    cv2.rectangle(mask, (22, 12), (234, 244), 255, -1)
    cv2.rectangle(mask, (12, 22), (244, 234), 255, -1)
    for x, y in [(22, 22), (234, 22), (22, 234), (234, 234)]:
        cv2.circle(mask, (x, y), 10, 255, -1)
    sy, sx = np.indices(mask.shape, dtype=np.float32)
    tile = face - 9 * sx / size - 6 * sy / size
    for px, py in PATTERNS[value]:
        cv2.circle(tile, (round(12 + px * 232), round(12 + py * 232)),
                   round(pip_radius * 232), 28, -1, cv2.LINE_AA)
    source = np.float32([[12, 12], [244, 12], [244, 244], [12, 244]])
    corners = np.float32([[-.5 + .16 * perspective, -.5], [.5, -.5 + .1 * perspective],
                          [.5 - .1 * perspective, .5], [-.5, .5 - .1 * perspective]])
    corners[:, 1] *= 1 - .25 * perspective
    theta = np.radians(angle)
    rotation = np.float32([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    target = (corners @ rotation.T * side + np.float32(center)) * 4
    transform = cv2.getPerspectiveTransform(source, target)
    # Recorte local para não alocar um frame inteiro em 4× para cada amostra.
    left, top = np.floor(target.min(axis=0) / 4 - 5).astype(int)
    right, bottom = np.ceil(target.max(axis=0) / 4 + 5).astype(int)
    transform[:2] -= np.array([[left * 4], [top * 4]]) * transform[2]
    shape = ((right - left) * 4, (bottom - top) * 4)
    warped_mask = cv2.warpPerspective(mask, transform, shape, flags=cv2.INTER_LINEAR)
    warped_tile = cv2.warpPerspective(tile, transform, shape, flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_REPLICATE)
    alpha = cv2.resize(warped_mask, (right - left, bottom - top), interpolation=cv2.INTER_AREA) / 255
    pixels = cv2.resize(warped_tile, (right - left, bottom - top), interpolation=cv2.INTER_AREA)
    roi = image[top:bottom, left:right]
    roi[:] = roi * (1 - alpha[:, :, None]) + pixels[:, :, None] * alpha[:, :, None]
    image = np.clip(image, 0, 255).astype(np.uint8)
    if blur:
        image[top:bottom, left:right] = cv2.GaussianBlur(image[top:bottom, left:right], (0, 0), blur)
    if noise:
        rng = np.random.default_rng(seed)
        image = np.clip(image.astype(np.float32) + rng.normal(0, noise, image.shape[:2])[:, :, None],
                        0, 255).astype(np.uint8)
    return image
