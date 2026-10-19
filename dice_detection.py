"""Isola faces claras e conta pontos escuros usando o pipeline clássico do PDF."""

from dataclasses import dataclass, field, replace
import math

import cv2
import numpy as np

from pipeline_parameters import CandidateFilters, PipFilters
from vision_pipeline import PreprocessingResult
from thresholding import threshold_image


@dataclass(frozen=True)
class Pip:
    center: tuple[float, float]
    radius: float
    area: float
    circularity: float


@dataclass(frozen=True)
class PointMeasurement:
    contour: np.ndarray  # Coordenadas locais do recorte.
    area: float
    perimeter: float
    relative_area: float
    aspect_ratio: float
    circularity: float
    rejection_reasons: tuple[str, ...]

    @property
    def accepted(self):
        return not self.rejection_reasons


@dataclass
class DiceCandidate:
    contour: np.ndarray
    bbox: tuple[int, int, int, int]
    area: float
    relative_area: float
    aspect_ratio: float
    extent: float
    face_bgr: np.ndarray
    pips: list[Pip]
    roi_bgr: np.ndarray | None = None
    roi_gray: np.ndarray | None = None
    face_mask: np.ndarray | None = None
    dark_mask: np.ndarray | None = None
    cleaned_mask: np.ndarray | None = None
    points_bgr: np.ndarray | None = None
    measurements: list[PointMeasurement] = field(default_factory=list)
    die_id: int | None = None
    solidity: float = 0.0
    rectangularity: float = 0.0
    label_below: bool = False

    @property
    def is_die(self):
        return 1 <= len(self.pips) <= 6

    @property
    def rejection_reason(self):
        if self.is_die:
            return ""
        return "Nenhum ponto válido." if not self.pips else "Mais de 6 pontos válidos."


@dataclass
class CandidateDetection:
    contour_count: int
    candidates: list[DiceCandidate]
    annotated_bgr: np.ndarray
    final_bgr: np.ndarray | None = None

    @property
    def dice(self):
        return [candidate for candidate in self.candidates if candidate.is_die]

    @property
    def dice_values(self):
        return [len(candidate.pips) for candidate in self.dice]

    @property
    def total_sum(self):
        return sum(self.dice_values)


def _find_pips(gray, face_mask, face_area, offset, filters):
    if not filters.normalize_scale:
        return _find_pips_native(gray, face_mask, face_area, offset, filters)
    # A ampliação interpola os níveis de cinza ANTES da segmentação. Medir
    # círculos de 2–3 px diretamente torna área e perímetro dependentes do
    # ângulo e da grade de pixels; ampliar uma máscara binária não resolve.
    scale = max(1, math.ceil(128 / min(gray.shape)))
    normalized_gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    normalized_mask = cv2.resize(face_mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
    pips, dark, cleaned, measurements = _find_pips_native(
        normalized_gray, normalized_mask, face_area * scale ** 2, (0, 0), filters,
    )
    pips = [replace(pip, center=((pip.center[0] + .5) / scale - .5 + offset[0],
                                (pip.center[1] + .5) / scale - .5 + offset[1]),
                    radius=pip.radius / scale, area=pip.area / scale ** 2) for pip in pips]
    measurements = [replace(point, contour=np.rint((point.contour + .5) / scale - .5).astype(np.int32),
                            area=point.area / scale ** 2, perimeter=point.perimeter / scale)
                    for point in measurements]
    size = (gray.shape[1], gray.shape[0])
    dark = cv2.resize(dark, size, interpolation=cv2.INTER_NEAREST)
    cleaned = cv2.resize(cleaned, size, interpolation=cv2.INTER_NEAREST)
    return pips, dark, cleaned, measurements


def _find_pips_native(gray, face_mask, face_area, offset, filters):
    dark = threshold_image(
        gray, filters.threshold_mode, filters.threshold,
        filters.adaptive_block_size, filters.adaptive_c, inverted=True, mask=face_mask,
    )
    dark = cv2.bitwise_and(dark, face_mask)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (filters.opening_kernel,) * 2)
    cleaned = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel, iterations=filters.opening_iterations)
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pips, measurements = [], []
    for contour in sorted(contours, key=lambda item: cv2.boundingRect(item)[1::-1]):
        perimeter = cv2.arcLength(contour, True)
        if filters.normalize_scale and perimeter > 0:
            # Pequenos degraus do raster não devem inflar o perímetro de um
            # ponto ampliado. A aproximação é relativa ao contorno e mantém
            # o formato geral; não completa pontos nem pressupõe sua contagem.
            contour = cv2.approxPolyDP(contour, .025 * perimeter, True)
            perimeter = cv2.arcLength(contour, True)
        area = cv2.contourArea(contour)
        _, _, width, height = cv2.boundingRect(contour)
        relative_area = area / face_area
        aspect_ratio = width / height
        circularity = 4 * math.pi * area / perimeter ** 2 if perimeter > 0 else 0.0
        reasons = []
        if perimeter <= 0:
            reasons.append("Perímetro nulo")
        if not filters.min_relative_area < relative_area < filters.max_relative_area:
            reasons.append("Área relativa fora dos limites")
        if not circularity > filters.min_circularity:
            reasons.append("Circularidade insuficiente")
        if not filters.min_aspect_ratio < aspect_ratio < filters.max_aspect_ratio:
            reasons.append("Proporção fora dos limites")
        measurements.append(PointMeasurement(
            contour, area, perimeter, relative_area, aspect_ratio, circularity, tuple(reasons),
        ))
        if not reasons:
            (cx, cy), radius = cv2.minEnclosingCircle(contour)
            pips.append(Pip((cx + offset[0], cy + offset[1]), radius, area, circularity))
    return pips, dark, cleaned, measurements


def _candidate_from_contour(result, contour, filters, pip_filters):
    area = cv2.contourArea(contour)
    x, y, width, height = cv2.boundingRect(contour)
    if area <= 0 or width <= 0 or height <= 0:
        return None
    relative_area = area / result.gray.size
    aspect_ratio = width / height
    extent = area / (width * height)
    hull_area = cv2.contourArea(cv2.convexHull(contour))
    solidity = area / hull_area if hull_area > 0 else 0.0
    rotated_width, rotated_height = cv2.minAreaRect(contour)[1]
    rotated_area = rotated_width * rotated_height
    rectangularity = area / rotated_area if rotated_area > 0 else 0.0
    if not (
        filters.min_relative_area < relative_area < filters.max_relative_area
        and filters.min_aspect_ratio < aspect_ratio < filters.max_aspect_ratio
        and extent > filters.min_extent
        and solidity > filters.min_solidity
        and rectangularity > filters.min_rectangularity
    ):
        return None
    mask_contour = cv2.convexHull(contour) if filters.convex_face_mask else contour
    local_contour = mask_contour - np.array([[[x, y]]], dtype=np.int32)
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
    gray = result.gray[y:y + height, x:x + width].copy()
    roi = result.resized_bgr[y:y + height, x:x + width].copy()
    pips, dark, cleaned, measurements = _find_pips(gray, mask, area, (x, y), pip_filters)
    points_image = roi.copy()
    for point in measurements:
        if point.accepted:
            cv2.drawContours(points_image, [point.contour], -1, (255, 100, 30), 2)
    candidate = DiceCandidate(
        contour, (x, y, width, height), area, relative_area, aspect_ratio, extent,
        cv2.bitwise_and(roi, roi, mask=mask), pips,
        roi, gray, mask, dark, cleaned, points_image, measurements, solidity=solidity,
        rectangularity=rectangularity,
    )
    return candidate


def _separate_touching_pair(result, contour, filters, pip_filters):
    """Divide um contorno com formato de duas faces quadradas justapostas.

    O eixo vem do retângulo girado, portanto também funciona na vertical e
    na diagonal. Só aceita a divisão se ambas as faces e seus pontos forem
    válidos e nenhum ponto for cortado pela fronteira entre as duas faces.
    """
    center, (width, height), angle = cv2.minAreaRect(contour)
    short, long = sorted((width, height))
    area = cv2.contourArea(contour)
    if short <= 0 or not 1.60 < long / short < 2.40:
        return []
    if not 2 * filters.min_relative_area < area / result.gray.size < 2 * filters.max_relative_area:
        return []
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    if (hull_area <= 0 or area / hull_area <= max(.85, filters.min_solidity)
            or area / (short * long) <= max(.80, filters.min_rectangularity)):
        return []
    radians = math.radians(angle + (90 if height > width else 0))
    axis = np.array([math.cos(radians), math.sin(radians)])
    x, y, roi_width, roi_height = cv2.boundingRect(contour)
    local_contour = contour - np.array([[[x, y]]], dtype=np.int32)
    mask = np.zeros((roi_height, roi_width), dtype=np.uint8)
    cv2.drawContours(mask, [local_contour], -1, 255, cv2.FILLED)
    yy, xx = np.indices(mask.shape)
    projection = (xx + x - center[0]) * axis[0] + (yy + y - center[1]) * axis[1]
    parts = []
    for side in (projection < 0, projection >= 0):
        part_mask = np.where(side, mask, 0).astype(np.uint8)
        part_contours, _ = cv2.findContours(part_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not part_contours:
            return []
        part = max(part_contours, key=cv2.contourArea) + np.array([[[x, y]]], dtype=np.int32)
        candidate = _candidate_from_contour(result, part, filters, pip_filters)
        if candidate is None or not candidate.is_die:
            return []
        parts.append(candidate)
    # Verifica os pontos no recorte unido para não transformar um ponto
    # atravessado pela divisão em dois pontos ou descartá-lo silenciosamente.
    point_mask = mask.copy()
    if filters.convex_face_mask:
        cv2.drawContours(point_mask, [hull - np.array([[[x, y]]], dtype=np.int32)], -1, 255, cv2.FILLED)
    gray = result.gray[y:y + roi_height, x:x + roi_width]
    pips, _, _, _ = _find_pips(gray, point_mask, area / 2, (x, y), pip_filters)
    if len(pips) != sum(len(part.pips) for part in parts):
        return []
    if any(abs(np.dot(np.array(pip.center) - center, axis)) <= pip.radius + 1 for pip in pips):
        return []
    # Os rótulos de faces vizinhas não devem se sobrepor.
    parts[1].label_below = True
    return parts


def detect_candidates(result: PreprocessingResult, filters=None, pip_filters=None) -> CandidateDetection:
    filters = filters or CandidateFilters()
    pip_filters = pip_filters or PipFilters()
    contours, _ = cv2.findContours(result.morphology, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    annotated = result.resized_bgr.copy()
    final = result.resized_bgr.copy()
    die_id = 0
    for contour in contours:
        parts = (_separate_touching_pair(result, contour, filters, pip_filters)
                 if filters.separate_touching_dice else [])
        if parts:
            candidates.extend(parts)
        else:
            candidate = _candidate_from_contour(result, contour, filters, pip_filters)
            if candidate is not None:
                candidates.append(candidate)
    candidates.sort(key=lambda item: item.bbox[:2])
    for candidate in candidates:
        x, y, _, _ = candidate.bbox
        contour, pips = candidate.contour, candidate.pips
        if candidate.is_die:
            die_id += 1
            candidate.die_id = die_id
        color = (80, 220, 80) if candidate.is_die else (0, 180, 255)
        label = f"Dado {die_id}: {len(pips)}" if candidate.is_die else f"Candidato: {len(pips)} pontos"
        for canvas in ([annotated, final] if candidate.is_die else [annotated]):
            cv2.drawContours(canvas, [contour], -1, color, 2)
            label_y = (min(canvas.shape[0] - 4, y + candidate.bbox[3] + 18)
                       if candidate.label_below else max(16, y - 6))
            cv2.putText(canvas, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for pip in pips:
            cv2.circle(annotated, tuple(round(v) for v in pip.center), round(pip.radius), (255, 100, 30), 1)
    return CandidateDetection(len(contours), candidates, annotated, final)
