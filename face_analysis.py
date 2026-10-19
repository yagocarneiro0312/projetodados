"""Inspeção de uma captura estável dos resultados, sem interromper a câmera."""

from copy import deepcopy

from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QAbstractItemView, QDialog, QDialogButtonBox, QHBoxLayout, QHeaderView,
    QLabel, QListWidget, QScrollArea, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from qt_image_utils import bgr_to_qimage, gray_to_qimage


class FaceAnalysisDialog(QDialog):
    def __init__(self, detection, parent=None):
        super().__init__(parent)
        self.detection = deepcopy(detection)
        self.setWindowTitle("Análise das faces — captura dos resultados")
        self.resize(1300, 700)
        self.setMinimumSize(750, 520)
        layout = QVBoxLayout(self)
        notice = QLabel("Captura dos resultados ao abrir esta janela. A câmera continua funcionando.")
        notice.setWordWrap(True)
        layout.addWidget(notice)
        content = QHBoxLayout()
        self.candidate_list = QListWidget()
        self.candidate_list.setMaximumWidth(240)
        self.candidate_list.setAccessibleName("Selecionar face para análise")
        for index, candidate in enumerate(self.detection.candidates, 1):
            text = (f"Dado {candidate.die_id}: {len(candidate.pips)}"
                    if candidate.is_die else f"Candidato {index}: rejeitado")
            self.candidate_list.addItem(text)
        content.addWidget(self.candidate_list)
        details = QVBoxLayout()
        self.summary = QLabel()
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        details.addWidget(self.summary)
        scroll = QScrollArea()
        scroll.setObjectName("analysisStages")
        scroll.viewport().setObjectName("analysisViewport")
        scroll.setWidgetResizable(True)
        scroll.setMinimumHeight(250)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        previews = QWidget()
        previews.setObjectName("analysisPreviews")
        row = QHBoxLayout(previews)
        self.image_labels = []
        for caption in ("Recorte original", "Máscara da face", "Pontos escuros", "Após abertura", "Pontos válidos"):
            column = QVBoxLayout()
            title = QLabel(caption)
            title.setAlignment(Qt.AlignmentFlag.AlignCenter)
            column.addWidget(title)
            label = QLabel()
            label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            label.setFixedSize(180, 180)
            column.addWidget(label)
            self.image_labels.append(label)
            row.addLayout(column)
        scroll.setWidget(previews)
        details.addWidget(scroll)
        self.table = QTableWidget(0, 7)
        self.table.setHorizontalHeaderLabels([
            "Ponto", "Área (px²)", "Perímetro (px)", "Área / face (%)",
            "Proporção", "Circularidade", "Validação",
        ])
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.table.horizontalHeader().setStretchLastSection(True)
        details.addWidget(self.table, 1)
        content.addLayout(details, 1)
        layout.addLayout(content, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.button(QDialogButtonBox.StandardButton.Close).setText("Fechar")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.candidate_list.currentRowChanged.connect(self.select_candidate)
        if self.detection.candidates:
            self.candidate_list.setCurrentRow(0)
        else:
            self.summary.setText("Nenhuma face passou pelos filtros geométricos nesta captura.")

    def select_candidate(self, index):
        if not 0 <= index < len(self.detection.candidates):
            return
        candidate = self.detection.candidates[index]
        status = (f"Dado {candidate.die_id}: {len(candidate.pips)} pontos válidos."
                  if candidate.is_die else f"Rejeitado: {candidate.rejection_reason}")
        self.summary.setText(
            f"{status}\nÁrea: {candidate.area:.1f} px² · Área relativa: {candidate.relative_area * 100:.4f}% · "
            f"Proporção: {candidate.aspect_ratio:.3f} · Extent: {candidate.extent:.3f} · Solidez: {candidate.solidity:.3f}\n"
            f"Retangularidade: {candidate.rectangularity:.3f}"
        )
        for index, (label, array) in enumerate(zip(self.image_labels, (
            candidate.roi_bgr, candidate.face_mask, candidate.dark_mask,
            candidate.cleaned_mask, candidate.points_bgr,
        ))):
            image = gray_to_qimage(array) if array.ndim == 2 else bgr_to_qimage(array)
            label.setPixmap(QPixmap.fromImage(image).scaled(
                label.size(), Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.FastTransformation if 1 <= index <= 3
                else Qt.TransformationMode.SmoothTransformation,
            ))
        self.table.setRowCount(len(candidate.measurements))
        for row, point in enumerate(candidate.measurements):
            values = [str(row + 1), f"{point.area:.2f}", f"{point.perimeter:.2f}",
                      f"{point.relative_area * 100:.4f}", f"{point.aspect_ratio:.3f}",
                      f"{point.circularity:.3f}",
                      "Aceito" if point.accepted else "; ".join(point.rejection_reasons)]
            for column, value in enumerate(values):
                self.table.setItem(row, column, QTableWidgetItem(value))
