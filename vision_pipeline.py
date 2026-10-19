"""Etapas de visão computacional para análise das faces de dados."""

from dataclasses import dataclass

import cv2
import numpy as np

from pipeline_parameters import PreprocessingParameters
from thresholding import threshold_image


STANDARD_WIDTH = 1000


@dataclass
class PreprocessingResult:
    resized_bgr: np.ndarray
    gray: np.ndarray
    original_size: tuple[int, int]
    blur: np.ndarray
    binary: np.ndarray
    closing: np.ndarray
    morphology: np.ndarray


def preprocess_frame(img: np.ndarray, parameters=None) -> PreprocessingResult:
    """Pré-processamento: largura padrão, cinza, threshold e morfologia.

    Recebe uma imagem BGR uint8, como as imagens utilizadas pelo OpenCV.
    Preserva a proporção tanto ao reduzir quanto ao ampliar a imagem.
    """
    if not isinstance(img, np.ndarray) or img.dtype != np.uint8:
        raise ValueError("A imagem deve ser um array uint8 em formato BGR.")
    if img.ndim != 3 or img.shape[2] != 3 or not img.size:
        raise ValueError("A imagem BGR deve ter altura, largura e três canais.")

    parameters = parameters or PreprocessingParameters()
    height, width = img.shape[:2]
    scale = parameters.width / width
    # fx/fy iguais reproduzem o PDF: dsize explícito alteraria ligeiramente
    # a escala vertical quando a altura calculada precisa ser arredondada.
    resized = (
        cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
        if round(height * scale) >= 1
        else cv2.resize(img, (parameters.width, 1), interpolation=cv2.INTER_LINEAR)
    )
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (parameters.blur_kernel,) * 2, 0)
    binary = threshold_image(
        blur, parameters.threshold_mode, parameters.face_threshold,
        parameters.adaptive_block_size, parameters.adaptive_c,
    )
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (parameters.morphology_kernel,) * 2)
    closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=parameters.closing_iterations)
    face_silhouettes = closing
    if parameters.fill_face_holes:
        # Os pontos são medidos no cinza original. Seus buracos na máscara da
        # face não devem fazer a abertura romper as pontes brancas entre eles.
        contours, _ = cv2.findContours(closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        face_silhouettes = closing.copy()
        for contour in contours:
            area = cv2.contourArea(contour)
            hull_area = cv2.contourArea(cv2.convexHull(contour))
            short, long = sorted(cv2.minAreaRect(contour)[1])
            # Desenhos do fundo podem formar laços grandes ao redor de um
            # dado. Preenchê-los engoliria a face antes da abertura separar
            # as linhas. Só completa regiões compactas de uma ou duas faces.
            if (short > 0 and 0 < area < .06 * gray.size and long / short < 2.5
                    and hull_area > 0 and area / hull_area > .8
                    and area / (short * long) > .7):
                cv2.drawContours(face_silhouettes, [contour], -1, 255, cv2.FILLED)
    morphology = cv2.morphologyEx(face_silhouettes, cv2.MORPH_OPEN, kernel, iterations=parameters.opening_iterations)
    return PreprocessingResult(
        resized_bgr=resized, gray=gray, original_size=(width, height),
        blur=blur, binary=binary, closing=closing, morphology=morphology,
    )
