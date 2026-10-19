"""Leitura de fotografias, incluindo nomes e diretórios Unicode no Windows."""

import cv2
import numpy as np


def load_bgr_image(path):
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None
    except (OSError, ValueError, cv2.error) as error:
        raise ValueError(f"Não foi possível ler a imagem: {error}") from error
    if image is None:
        raise ValueError("Arquivo vazio, inválido ou formato de imagem não suportado.")
    return image
