"""Conversões entre as imagens do Qt e os arrays do OpenCV."""

import numpy as np
from PyQt6.QtGui import QImage


def qimage_to_bgr(image: QImage) -> np.ndarray:
    """Copia os pixels em ordem BGR, respeitando o alinhamento das linhas."""
    if image.isNull():
        raise ValueError("A imagem da câmera está vazia.")
    bgr = image.convertToFormat(QImage.Format.Format_BGR888)
    buffer = bgr.constBits()
    buffer.setsize(bgr.sizeInBytes())
    rows = np.frombuffer(buffer, dtype=np.uint8).reshape(bgr.height(), bgr.bytesPerLine())
    return rows[:, :bgr.width() * 3].reshape(bgr.height(), bgr.width(), 3).copy()


def gray_to_qimage(gray: np.ndarray) -> QImage:
    """Retorna uma imagem que mantém seus pixels após o array ser liberado."""
    if gray.ndim != 2 or gray.dtype != np.uint8 or not gray.size:
        raise ValueError("A imagem em cinza deve ser um array uint8 com dois eixos.")
    gray = np.ascontiguousarray(gray)
    height, width = gray.shape
    return QImage(
        gray.data, width, height, gray.strides[0], QImage.Format.Format_Grayscale8,
    ).copy()


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    """Converte a imagem colorida com anotações, mantendo seus pixels próprios."""
    bgr = np.ascontiguousarray(bgr, dtype=np.uint8)
    height, width = bgr.shape[:2]
    return QImage(
        bgr.data, width, height, bgr.strides[0], QImage.Format.Format_BGR888,
    ).copy()
