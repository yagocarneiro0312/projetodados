"""Edição conjunta dos parâmetros de preparação, faces e pontos."""

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QDoubleSpinBox, QFormLayout, QLabel, QSpinBox,
    QTabWidget, QVBoxLayout, QWidget,
)

from pipeline_parameters import CandidateFilters, PipFilters, PreprocessingParameters, THRESHOLD_MODES, c920_calibration, emeet_calibration


class CandidateSettingsDialog(QDialog):
    def __init__(self, filters, parent=None, parameters=None, pip_filters=None):
        super().__init__(parent)
        self.setWindowTitle("Parâmetros do detector")
        self.setMinimumWidth(640)
        layout = QVBoxLayout(self)
        description = QLabel(
            "Limites geométricos são estritos. A área das faces é relativa à imagem; "
            "a área dos pontos é relativa à face. Kernels devem ser ímpares."
        )
        description.setWordWrap(True)
        layout.addWidget(description)
        tabs = QTabWidget()
        self.preparation_fields = {}
        self.fields = {}  # Mantém acesso aos filtros de candidatos existentes.
        self.pip_fields = {}
        self._add_tab(tabs, "Preparação", self.preparation_fields, [
            ("width", "Largura (px)", 1, 10000, None, 100),
            ("blur_kernel", "Kernel Gaussian (N × N)", 1, 99, None, 2),
            ("threshold_mode", "Modo de threshold das faces", 0, 0, "mode", 0),
            ("face_threshold", "Threshold das faces", 0, 255, None, 1),
            ("adaptive_block_size", "Bloco adaptativo (N × N)", 3, 501, None, 2),
            ("adaptive_c", "Constante C", -255, 255, 2, 1),
            ("morphology_kernel", "Kernel elíptico das faces", 1, 99, None, 2),
            ("closing_iterations", "Iterações do fechamento", 0, 20, None, 1),
            ("opening_iterations", "Iterações da abertura", 0, 20, None, 1),
            ("fill_face_holes", "Preencher face antes da abertura", 0, 0, "bool", 0),
        ])
        self._add_tab(tabs, "Faces", self.fields, [
            ("min_relative_area", "Área mínima (%)", 0, 100, 4, 0.01),
            ("max_relative_area", "Área máxima (%)", 0, 100, 4, 0.1),
            ("min_aspect_ratio", "Proporção mínima (w/h)", 0.01, 20, 2, 0.05),
            ("max_aspect_ratio", "Proporção máxima (w/h)", 0.01, 20, 2, 0.05),
            ("min_extent", "Extent mínimo", 0, 1, 3, 0.05),
            ("min_solidity", "Solidez mínima", 0, 1, 3, 0.05),
            ("min_rectangularity", "Retangularidade mínima", 0, 1, 3, 0.05),
            ("separate_touching_dice", "Separar dados encostados", 0, 0, "bool", 0),
            ("convex_face_mask", "Preencher máscara convexa", 0, 0, "bool", 0),
        ])
        self._add_tab(tabs, "Pontos", self.pip_fields, [
            ("threshold_mode", "Modo de threshold dos pontos", 0, 0, "mode", 0),
            ("threshold", "Threshold invertido dos pontos", 0, 255, None, 1),
            ("adaptive_block_size", "Bloco adaptativo (N × N)", 3, 501, None, 2),
            ("adaptive_c", "Constante C", -255, 255, 2, 1),
            ("opening_kernel", "Kernel elíptico dos pontos", 1, 99, None, 2),
            ("opening_iterations", "Iterações da abertura", 0, 20, None, 1),
            ("normalize_scale", "Normalizar escala dos pontos", 0, 0, "bool", 0),
            ("min_relative_area", "Área mínima / face (%)", 0, 100, 4, 0.1),
            ("max_relative_area", "Área máxima / face (%)", 0, 100, 4, 0.1),
            ("min_circularity", "Circularidade mínima", 0, 1, 3, 0.05),
            ("min_aspect_ratio", "Proporção mínima (w/h)", 0.01, 20, 2, 0.05),
            ("max_aspect_ratio", "Proporção máxima (w/h)", 0.01, 20, 2, 0.05),
        ])
        layout.addWidget(tabs)
        self.error_label = QLabel()
        self.error_label.setTextFormat(Qt.TextFormat.PlainText)
        self.error_label.setWordWrap(True)
        layout.addWidget(self.error_label)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Aplicar")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("Cancelar")
        reset = buttons.addButton("Restaurar padrões do PDF", QDialogButtonBox.ButtonRole.ResetRole)
        reset.clicked.connect(self.restore_defaults)
        emeet = buttons.addButton("Perfil EMEET", QDialogButtonBox.ButtonRole.ResetRole)
        emeet.setToolTip("Repor os parâmetros para a EMEET S600 em 1280 × 720 sobre fundo estampado.")
        emeet.clicked.connect(self.restore_emeet_calibration)
        calibration = buttons.addButton("Perfil C920", QDialogButtonBox.ButtonRole.ResetRole)
        calibration.setToolTip("Repor os parâmetros calibrados para as faces e pontos na C920.")
        calibration.clicked.connect(self.restore_calibration)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.set_filters(filters)
        self._set_values(self.preparation_fields, parameters or PreprocessingParameters())
        self._set_values(self.pip_fields, pip_filters or PipFilters())
        for fields in (self.preparation_fields, self.pip_fields):
            fields["threshold_mode"].currentIndexChanged.connect(
                lambda _, group=fields: self._update_threshold_controls(group)
            )
            self._update_threshold_controls(fields)

    @staticmethod
    def _update_threshold_controls(fields):
        mode = fields["threshold_mode"].currentData()
        adaptive = mode in ("adaptive_gaussian", "adaptive_mean")
        fields["adaptive_block_size"].setEnabled(adaptive)
        fields["adaptive_c"].setEnabled(adaptive)
        fields["face_threshold" if "face_threshold" in fields else "threshold"].setEnabled(mode == "fixed")

    @staticmethod
    def _add_tab(tabs, title, fields, specs):
        page = QWidget()
        form = QFormLayout(page)
        for name, label, minimum, maximum, decimals, step in specs:
            if decimals == "bool":
                field = QCheckBox("Ativar")
            elif decimals == "mode":
                field = QComboBox()
                for mode, caption in THRESHOLD_MODES.items():
                    field.addItem(caption, mode)
            else:
                field = QSpinBox() if decimals is None else QDoubleSpinBox()
                field.setRange(minimum, maximum)
                if decimals is not None:
                    field.setDecimals(decimals)
                field.setSingleStep(step)
            if name == "adaptive_block_size":
                field.setToolTip("Região local ímpar, maior que 1, usada para calcular o threshold.")
            elif name == "adaptive_c":
                field.setToolTip("C é subtraído da média local. Faces claras: comece com −10; pontos escuros: +7.")
            elif name == "min_solidity":
                field.setToolTip("Área / área da envoltória convexa. Rejeita regiões irregulares; zero desativa o filtro.")
            elif name == "convex_face_mask":
                field.setToolTip("Preenche pequenas reentrâncias na máscara da face para preservar pontos próximos das bordas.")
            elif name == "min_rectangularity":
                field.setToolTip("Área / área do menor retângulo girado. Rejeita desenhos do fundo e aceita faces giradas; zero desativa.")
            elif name == "separate_touching_dice":
                field.setToolTip("Separa pares de faces claras unidas quando as duas partes têm formato e pontos válidos.")
            elif name == "fill_face_holes":
                field.setToolTip("Preenche apenas buracos internos antes da abertura, para não fragmentar a face entre os pontos.")
            elif name == "normalize_scale":
                field.setToolTip("Amplia o cinza da face para pelo menos 128 px antes de medir os pontos. O kernel dos pontos usa essa escala.")
            field.setAccessibleName(label)
            fields[name] = field
            form.addRow(label, field)
        tabs.addTab(page, title)

    @staticmethod
    def _set_values(fields, parameters):
        for name, field in fields.items():
            value = getattr(parameters, name)
            if isinstance(field, QCheckBox):
                field.setChecked(value)
            elif isinstance(field, QComboBox):
                field.setCurrentIndex(field.findData(value))
            else:
                field.setValue(value * 100 if "relative_area" in name else value)

    @staticmethod
    def _values(fields):
        return {name: field.isChecked() if isinstance(field, QCheckBox)
                else field.currentData() if isinstance(field, QComboBox)
                else field.value() / 100 if "relative_area" in name else field.value()
                for name, field in fields.items()}

    def set_filters(self, filters):
        self._set_values(self.fields, filters)
        self.error_label.clear()

    def restore_defaults(self):
        self.set_filters(CandidateFilters())
        self._set_values(self.preparation_fields, PreprocessingParameters())
        self._set_values(self.pip_fields, PipFilters())

    def restore_calibration(self):
        parameters, filters, pip_filters = c920_calibration()
        self.set_filters(filters)
        self._set_values(self.preparation_fields, parameters)
        self._set_values(self.pip_fields, pip_filters)

    def restore_emeet_calibration(self):
        parameters, filters, pip_filters = emeet_calibration()
        self.set_filters(filters)
        self._set_values(self.preparation_fields, parameters)
        self._set_values(self.pip_fields, pip_filters)

    def selected_filters(self):
        return CandidateFilters(**self._values(self.fields))

    def selected_parameters(self):
        return PreprocessingParameters(**self._values(self.preparation_fields))

    def selected_pip_filters(self):
        return PipFilters(**self._values(self.pip_fields))

    def selected_configuration(self):
        return self.selected_parameters(), self.selected_filters(), self.selected_pip_filters()

    def accept(self):
        try:
            self.selected_configuration()
        except ValueError as error:
            self.error_label.setText(str(error))
            return
        super().accept()
