"""Parâmetros ajustáveis e padrões do detector clássico descrito no PDF."""

from dataclasses import dataclass, replace
import math


THRESHOLD_MODES = {
    "fixed": "Fixo (PDF)",
    "adaptive_gaussian": "Adaptativo gaussiano",
    "adaptive_mean": "Adaptativo pela média",
    "otsu": "Otsu (automático)",
}


def _threshold_parameters(mode, block_size, constant):
    if mode not in THRESHOLD_MODES:
        raise ValueError("Modo de threshold inválido.")
    _integer(block_size, "Bloco adaptativo", 3, odd=True)
    if not math.isfinite(constant):
        raise ValueError("A constante C deve ser finita.")


def _integer(value, label, minimum, maximum=None, odd=False):
    if (type(value) is not int or value < minimum
            or (maximum is not None and value > maximum) or (odd and value % 2 == 0)):
        raise ValueError(f"{label}: informe um inteiro válido" + (" ímpar." if odd else "."))


def _limits(minimum, maximum, label, upper=None):
    if not all(math.isfinite(v) for v in (minimum, maximum)):
        raise ValueError(f"{label}: os limites precisam ser finitos.")
    if not 0 <= minimum < maximum or (upper is not None and maximum > upper):
        raise ValueError(f"{label}: o mínimo deve ser menor que o máximo e estar no intervalo permitido.")


@dataclass(frozen=True)
class PreprocessingParameters:
    width: int = 1000
    blur_kernel: int = 5
    face_threshold: int = 190
    morphology_kernel: int = 5
    closing_iterations: int = 2
    opening_iterations: int = 1
    threshold_mode: str = "fixed"
    adaptive_block_size: int = 101
    adaptive_c: float = -10.0
    fill_face_holes: bool = False

    def __post_init__(self):
        _integer(self.width, "Largura", 1)
        _integer(self.blur_kernel, "Kernel Gaussian", 1, odd=True)
        _integer(self.face_threshold, "Threshold das faces", 0, 255)
        _integer(self.morphology_kernel, "Kernel das faces", 1, odd=True)
        _integer(self.closing_iterations, "Iterações do fechamento", 0)
        _integer(self.opening_iterations, "Iterações da abertura", 0)
        _threshold_parameters(self.threshold_mode, self.adaptive_block_size, self.adaptive_c)
        if type(self.fill_face_holes) is not bool:
            raise ValueError("A opção de preencher faces deve ser booleana.")


@dataclass(frozen=True)
class CandidateFilters:
    min_relative_area: float = 0.0005
    max_relative_area: float = 0.03
    min_aspect_ratio: float = 0.55
    max_aspect_ratio: float = 1.60
    min_extent: float = 0.35
    min_solidity: float = 0.0
    convex_face_mask: bool = False
    min_rectangularity: float = 0.0
    separate_touching_dice: bool = False

    def __post_init__(self):
        _limits(self.min_relative_area, self.max_relative_area, "Área das faces", 1)
        _limits(self.min_aspect_ratio, self.max_aspect_ratio, "Proporção das faces")
        if self.min_aspect_ratio <= 0:
            raise ValueError("A proporção mínima deve ser positiva.")
        if not math.isfinite(self.min_extent) or not 0 <= self.min_extent <= 1:
            raise ValueError("O extent deve estar entre 0 e 1 e ser finito.")
        if not math.isfinite(self.min_solidity) or not 0 <= self.min_solidity <= 1:
            raise ValueError("A solidez deve estar entre 0 e 1 e ser finita.")
        if type(self.convex_face_mask) is not bool:
            raise ValueError("A opção de máscara convexa deve ser booleana.")
        if not math.isfinite(self.min_rectangularity) or not 0 <= self.min_rectangularity <= 1:
            raise ValueError("A retangularidade deve estar entre 0 e 1 e ser finita.")
        if type(self.separate_touching_dice) is not bool:
            raise ValueError("A opção de separar dados encostados deve ser booleana.")


@dataclass(frozen=True)
class PipFilters:
    threshold: int = 130
    opening_kernel: int = 3
    opening_iterations: int = 1
    min_relative_area: float = 0.004
    max_relative_area: float = 0.10
    min_circularity: float = 0.55
    min_aspect_ratio: float = 0.60
    max_aspect_ratio: float = 1.50
    threshold_mode: str = "fixed"
    adaptive_block_size: int = 31
    adaptive_c: float = 7.0
    normalize_scale: bool = False

    def __post_init__(self):
        _integer(self.threshold, "Threshold dos pontos", 0, 255)
        _integer(self.opening_kernel, "Kernel dos pontos", 1, odd=True)
        _integer(self.opening_iterations, "Iterações da abertura dos pontos", 0)
        _threshold_parameters(self.threshold_mode, self.adaptive_block_size, self.adaptive_c)
        if type(self.normalize_scale) is not bool:
            raise ValueError("A opção de normalizar pontos deve ser booleana.")
        _limits(self.min_relative_area, self.max_relative_area, "Área dos pontos", 1)
        _limits(self.min_aspect_ratio, self.max_aspect_ratio, "Proporção dos pontos")
        if self.min_aspect_ratio <= 0:
            raise ValueError("A proporção mínima deve ser positiva.")
        if not math.isfinite(self.min_circularity) or not 0 <= self.min_circularity <= 1:
            raise ValueError("A circularidade mínima deve estar entre 0 e 1 e ser finita.")


def c920_calibration():
    """Perfil medido com a C920 em 1920×1080, zoom 1× e largura de análise 1000.

    Os limites continuam aceitando valores de 1 a 6; não usam a contagem
    esperada da cena para decidir o resultado.
    """
    return (
        PreprocessingParameters(threshold_mode="adaptive_gaussian"),
        CandidateFilters(min_relative_area=0.001, max_relative_area=0.03,
                         min_aspect_ratio=0.70, max_aspect_ratio=1.40,
                         min_extent=0.45, min_solidity=0.85, convex_face_mask=True),
        PipFilters(threshold_mode="adaptive_gaussian", min_relative_area=0.008,
                   max_relative_area=0.075, min_circularity=0.65),
    )


def emeet_calibration():
    """Perfil para faces pequenas na EMEET S600 em 1280×720 sobre fundo estampado.

    Evita unir as faces aos traços claros do fundo durante o fechamento.
    Preenche buracos de regiões compactas antes da abertura. Os pontos são
    medidos em escala normalizada, com Otsu dentro da face e circularidade 0,70.
    """
    parameters, faces, pips = c920_calibration()
    return (
        replace(parameters, blur_kernel=3, closing_iterations=0, fill_face_holes=True),
        replace(faces, min_rectangularity=0.80, separate_touching_dice=True),
        replace(pips, min_circularity=0.70, threshold_mode="otsu", normalize_scale=True),
    )
