"""Miniatura que permite inspecionar sua etapa ao passar o mouse."""

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import QFrame, QLabel, QVBoxLayout


class StagePreview(QFrame):
    hovered = pyqtSignal(int, bool)

    def __init__(self, index, caption, parent=None):
        super().__init__(parent)
        self.index = index
        self.setObjectName("stageCard")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Passe o mouse para ver esta etapa na imagem principal.")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)
        self.caption = QLabel(caption)
        self.caption.setObjectName("stageCaption")
        self.caption.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.caption)
        self.image_preview = QLabel("Aguardando imagem")
        self.image_preview.setObjectName("stagePreview")
        self.image_preview.setFixedSize(156, 80)
        self.image_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.image_preview, 0, Qt.AlignmentFlag.AlignHCenter)

    def enterEvent(self, event):
        self.hovered.emit(self.index, True)
        super().enterEvent(event)

    def leaveEvent(self, event):
        self.hovered.emit(self.index, False)
        super().leaveEvent(event)

    def hideEvent(self, event):
        self.hovered.emit(self.index, False)
        super().hideEvent(event)
