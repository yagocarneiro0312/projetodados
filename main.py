"""Interface para webcam e fotografias com o detector clássico de dados."""

import sys
import time
from pathlib import Path

import cv2
from PyQt6.QtCore import QPointF, QRectF, Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QCloseEvent, QFont, QImage, QPainter, QPen, QPixmap
from PyQt6.QtMultimedia import (
    QCamera, QMediaCaptureSession, QMediaDevices, QVideoFrame, QVideoSink,
)
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from candidate_settings import CandidateSettingsDialog
from dice_detection import CandidateFilters, detect_candidates
from dice_chart import DiceChartPanel
from face_analysis import FaceAnalysisDialog
from image_source import load_bgr_image
from pipeline_parameters import emeet_calibration
from qt_image_utils import bgr_to_qimage, gray_to_qimage, qimage_to_bgr
from stage_preview import StagePreview
from vision_pipeline import preprocess_frame


DEFAULT_CAMERA_NAME = "EMEET SmartCam S600"
DEFAULT_CAMERA_QUALITY = "1280 × 720 · 60 fps"


class VideoPreview(QWidget):
    """Exibe frames com zoom central e brilho, sem depender do hardware."""

    frame_received = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.image = QImage()
        self.zoom = 1.0
        self.brightness = 0
        self.receiving = False
        self.detection = None
        self.stage_image = QImage()
        self.stage_title = ""
        self.stage_binary = False
        self.sink = QVideoSink(self)
        self.sink.videoFrameChanged.connect(self.receive_frame)

    def videoSink(self):
        return self.sink

    def receive_frame(self, frame):
        if self.receiving and frame.isValid():
            image = frame.toImage()
            if not image.isNull():
                self.image = image
                self.update()
                self.frame_received.emit()

    def clear(self):
        self.receiving = False
        self.image = QImage()
        self.detection = None
        self.set_stage_image()
        self.sink.setVideoFrame(QVideoFrame())
        self.update()

    def set_zoom(self, value):
        self.zoom = value / 100
        self.detection = None
        self.update()

    def set_stage_image(self, image=None, title="", binary=False):
        self.stage_image = QImage() if image is None else image
        self.stage_title = title
        self.stage_binary = binary
        self.update()

    def set_brightness(self, value):
        self.brightness = value
        self.detection = None
        self.update()

    def adjusted_image(self):
        """Imagem da região exibida, com os mesmos ajustes de zoom e brilho."""
        if self.image.isNull():
            return QImage()
        width = max(1, round(self.image.width() / self.zoom))
        height = max(1, round(self.image.height() / self.zoom))
        image = self.image.copy(
            (self.image.width() - width) // 2,
            (self.image.height() - height) // 2,
            width, height,
        )
        if self.brightness:
            painter = QPainter(image)
            color = QColor("white" if self.brightness > 0 else "black")
            color.setAlphaF(abs(self.brightness) / 100)
            painter.fillRect(image.rect(), color)
            painter.end()
        return image

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#050608"))
        if not self.stage_image.isNull():
            # A etapa já incorpora os ajustes da fonte. Exibir sem reaplicar
            # zoom/brilho ou desenhar os contornos da câmera sobre uma máscara.
            painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, not self.stage_binary)
            width, height = self.stage_image.width(), self.stage_image.height()
            scale = min(self.width() / width, self.height() / height)
            target = QRectF((self.width() - width * scale) / 2,
                            (self.height() - height * scale) / 2, width * scale, height * scale)
            painter.drawImage(target, self.stage_image)
            painter.setBackground(QColor("#050608"))
            painter.setBackgroundMode(Qt.BGMode.OpaqueMode)
            painter.setPen(QColor("#86efac"))
            painter.drawText(QPointF(12, 24), self.stage_title)
            return
        if self.image.isNull():
            return

        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        width = self.image.width() / self.zoom
        height = self.image.height() / self.zoom
        source = QRectF(
            (self.image.width() - width) / 2,
            (self.image.height() - height) / 2,
            width, height,
        )
        scale = min(self.width() / width, self.height() / height)
        target = QRectF(
            (self.width() - width * scale) / 2,
            (self.height() - height * scale) / 2,
            width * scale, height * scale,
        )
        painter.drawImage(target, self.image, source)
        if self.brightness:
            color = QColor("white" if self.brightness > 0 else "black")
            color.setAlphaF(abs(self.brightness) / 100)
            painter.fillRect(target, color)

        if self.detection is not None:
            image_height, image_width = self.detection.annotated_bgr.shape[:2]
            scale_x = target.width() / image_width
            scale_y = target.height() / image_height
            painter.setClipRect(target)
            painter.setBackground(QColor("#050608"))
            painter.setBackgroundMode(Qt.BGMode.OpaqueMode)
            for candidate in self.detection.candidates:
                color = QColor("#50dc50" if candidate.is_die else "#ffb400")
                painter.setPen(QPen(color, 2))
                x, y, width, height = candidate.bbox
                left = target.left() + x * scale_x
                top = target.top() + y * scale_y
                painter.drawRect(QRectF(left, top, width * scale_x, height * scale_y))
                text = (
                    f"Dado {candidate.die_id}: {len(candidate.pips)}"
                    if candidate.is_die else f"Candidato: {len(candidate.pips)} pontos"
                )
                label_y = (min(target.bottom() - 4, top + height * scale_y + painter.fontMetrics().height())
                           if candidate.label_below else max(target.top() + 16, top - 5))
                painter.drawText(QPointF(left, label_y), text)
                painter.setPen(QPen(QColor("#1e64ff"), 1))
                for pip in candidate.pips:
                    center_x = target.left() + pip.center[0] * scale_x
                    center_y = target.top() + pip.center[1] * scale_y
                    painter.drawEllipse(QRectF(
                        center_x - pip.radius * scale_x, center_y - pip.radius * scale_y,
                        pip.radius * 2 * scale_x, pip.radius * 2 * scale_y,
                    ))


class WebcamWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Projeto Dados — Detector")
        self.resize(1440, 990)
        self.setMinimumSize(900, 920)

        self.camera = None
        self.devices = []
        self.media_devices = QMediaDevices(self)
        self.capture_session = QMediaCaptureSession(self)
        self.reopen_timer = QTimer(self)
        self.reopen_timer.setSingleShot(True)
        self.reopen_timer.setInterval(100)
        self.reopen_timer.timeout.connect(self.open_camera)
        self.start_timer = QTimer(self)
        self.start_timer.setSingleShot(True)
        self.start_timer.setInterval(10000)
        self.start_timer.timeout.connect(self.camera_timed_out)
        self.pipeline_timer = QTimer(self)
        self.pipeline_timer.setInterval(100)
        self.pipeline_timer.timeout.connect(self.process_current_frame)
        self.pipeline_result = None
        self.detection_result = None
        self.hovered_stage = None
        self.stage_images = []
        self.preprocessing_parameters, self.candidate_filters, self.pip_filters = emeet_calibration()
        self.source = "camera"
        self.photo_path = None
        self.photo_image = QImage()
        self._photo_fallback = False
        self._awaiting_camera_frame = False
        self.startup_timer = QTimer(self)
        self.startup_timer.setSingleShot(True)
        self.startup_timer.timeout.connect(self.open_preferred_camera)

        central = QWidget()
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(24, 24, 24, 24)
        layout.setSpacing(8)
        camera_column = QVBoxLayout()
        camera_column.setSpacing(8)
        sidebar_column = QVBoxLayout()
        sidebar_column.setSpacing(8)
        layout.addLayout(camera_column, 1)
        layout.addLayout(sidebar_column)

        self.camera_panel = QFrame()
        self.camera_panel.setObjectName("cameraPanel")
        camera_layout = QVBoxLayout(self.camera_panel)
        camera_layout.setContentsMargins(4, 4, 4, 4)

        self.preview = QStackedWidget()
        self.placeholder = QLabel("A webcam está fechada.\n\nSelecione uma câmera no painel ao lado.")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.placeholder.setObjectName("placeholder")
        self.preview.addWidget(self.placeholder)
        self.video_widget = VideoPreview()
        self.video_widget.frame_received.connect(self.camera_frame_received)
        self.preview.addWidget(self.video_widget)
        self.capture_session.setVideoSink(self.video_widget.videoSink())
        camera_layout.addWidget(self.preview)
        camera_column.addWidget(self.camera_panel, 1)

        self.pipeline_panel = QFrame()
        self.pipeline_panel.setObjectName("pipelinePanel")
        self.pipeline_panel.setFixedHeight(256)
        pipeline_layout = QVBoxLayout(self.pipeline_panel)
        pipeline_layout.setContentsMargins(16, 12, 16, 12)
        pipeline_layout.setSpacing(8)
        pipeline_header = QHBoxLayout()
        pipeline_title = QLabel("Pipeline de detecção")
        pipeline_title.setObjectName("pipelineTitle")
        pipeline_title.setWordWrap(True)
        pipeline_header.addWidget(pipeline_title, 1)
        self.filters_button = QPushButton("Filtros")
        self.filters_button.clicked.connect(self.configure_candidate_filters)
        pipeline_header.addWidget(self.filters_button)
        self.analysis_button = QPushButton("Análise das faces")
        self.analysis_button.setEnabled(False)
        self.analysis_button.clicked.connect(self.show_face_analysis)
        pipeline_header.addWidget(self.analysis_button)
        self.pipeline_enabled = QCheckBox("Processar")
        self.pipeline_enabled.setChecked(True)
        self.pipeline_enabled.toggled.connect(self.toggle_pipeline)
        pipeline_header.addWidget(self.pipeline_enabled)
        pipeline_layout.addLayout(pipeline_header)

        self.pipeline_scroll = QScrollArea()
        self.pipeline_scroll.setObjectName("pipelineStages")
        self.pipeline_scroll.viewport().setObjectName("stageViewport")
        self.pipeline_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.pipeline_scroll.setWidgetResizable(True)
        self.pipeline_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.pipeline_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.preview_content = QWidget()
        self.preview_content.setObjectName("stagePreviews")
        preview_row = QHBoxLayout(self.preview_content)
        preview_row.setContentsMargins(0, 0, 0, 0)
        preview_row.setSpacing(12)
        self.stage_previews = []
        self.stage_captions = []
        self.stage_cards = []
        for index, caption in enumerate([
            "Colorida · 1000 px", "Cinza · 1000 px", "Gaussian Blur 5×5", "Threshold > 190",
            "Fechamento · 2×", "Abertura · 1×",
            "Candidatos e dados", "Resultado final",
        ]):
            card = StagePreview(index, caption)
            card.hovered.connect(self.hover_stage)
            preview_row.addWidget(card, 1)
            self.stage_cards.append(card)
            self.stage_captions.append(card.caption)
            self.stage_previews.append(card.image_preview)
        (
            self.color_preview, self.gray_preview, self.blur_preview, self.binary_preview,
            self.closing_preview, self.morphology_preview,
            self.candidates_preview, self.final_preview,
        ) = self.stage_previews
        self.closing_preview.setToolTip("Fechamento com kernel elíptico 5×5 e 2 iterações.")
        self.morphology_preview.setToolTip("Abertura com kernel elíptico 5×5 e 1 iteração.")
        self.candidates_preview.setToolTip("Verde: dado com 1–6 pontos. Amarelo: candidato não validado.")
        self.final_preview.setToolTip("Somente os dados aceitos, numerados da esquerda para a direita.")
        self.pipeline_scroll.setWidget(self.preview_content)
        pipeline_layout.addWidget(self.pipeline_scroll, 1)
        self.pipeline_info = QLabel("Abra a webcam para iniciar o processamento.")
        self.pipeline_info.setObjectName("pipelineInfo")
        self.pipeline_info.setWordWrap(True)
        self.pipeline_info.setMinimumHeight(36)
        self.pipeline_info.setTextFormat(Qt.TextFormat.PlainText)
        pipeline_layout.addWidget(self.pipeline_info)
        camera_column.addWidget(self.pipeline_panel)

        self.settings_panel = QFrame()
        self.settings_panel.setObjectName("settingsPanel")
        self.settings_panel.setFixedWidth(280)
        settings_container = QVBoxLayout(self.settings_panel)
        settings_container.setContentsMargins(2, 2, 2, 2)
        self.settings_scroll = QScrollArea()
        self.settings_scroll.setObjectName("settingsScroll")
        self.settings_scroll.viewport().setObjectName("settingsViewport")
        self.settings_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.settings_scroll.setWidgetResizable(True)
        self.settings_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        settings_content = QWidget()
        settings_content.setObjectName("settingsContent")
        settings = QVBoxLayout(settings_content)
        settings.setContentsMargins(18, 16, 18, 16)
        settings.setSpacing(10)
        title = QLabel("Configurações")
        title.setObjectName("title")
        settings.addWidget(title)

        camera_label = QLabel("Câmera")
        settings.addWidget(camera_label)
        self.camera_combo = QComboBox()
        self.camera_combo.setAccessibleName("Selecionar webcam")
        self.camera_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        self.camera_combo.setMinimumContentsLength(10)
        camera_label.setBuddy(self.camera_combo)
        settings.addWidget(self.camera_combo)
        self.refresh_button = QPushButton("Atualizar lista")
        self.refresh_button.clicked.connect(self.refresh_devices)
        settings.addWidget(self.refresh_button)

        quality_label = QLabel("Qualidade / resolução")
        settings.addWidget(quality_label)
        self.quality_combo = QComboBox()
        self.quality_combo.setAccessibleName("Qualidade da câmera")
        quality_label.setBuddy(self.quality_combo)
        self.quality_combo.currentIndexChanged.connect(self.quality_changed)
        settings.addWidget(self.quality_combo)

        zoom_header = QHBoxLayout()
        zoom_label = QLabel("Zoom")
        zoom_header.addWidget(zoom_label)
        self.zoom_value = QLabel("1,0×")
        self.zoom_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        zoom_header.addWidget(self.zoom_value)
        settings.addLayout(zoom_header)
        self.zoom_slider = QSlider(Qt.Orientation.Horizontal)
        self.zoom_slider.setRange(100, 400)
        self.zoom_slider.setValue(100)
        self.zoom_slider.setAccessibleName("Zoom digital")
        zoom_label.setBuddy(self.zoom_slider)
        self.zoom_slider.valueChanged.connect(self.zoom_changed)
        settings.addWidget(self.zoom_slider)

        brightness_header = QHBoxLayout()
        brightness_label = QLabel("Brilho")
        brightness_header.addWidget(brightness_label)
        self.brightness_value = QLabel("0")
        self.brightness_value.setAlignment(Qt.AlignmentFlag.AlignRight)
        brightness_header.addWidget(self.brightness_value)
        settings.addLayout(brightness_header)
        self.brightness_slider = QSlider(Qt.Orientation.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setAccessibleName("Brilho da imagem")
        brightness_label.setBuddy(self.brightness_slider)
        self.brightness_slider.valueChanged.connect(self.brightness_changed)
        settings.addWidget(self.brightness_slider)
        self.reset_button = QPushButton("Restaurar zoom e brilho")
        self.reset_button.clicked.connect(self.reset_adjustments)
        settings.addWidget(self.reset_button)
        settings.addStretch()

        self.open_button = QPushButton("Abrir webcam")
        self.open_button.setObjectName("primary")
        self.open_button.clicked.connect(self.open_camera)
        settings.addWidget(self.open_button)
        self.close_button = QPushButton("Fechar webcam")
        self.close_button.clicked.connect(self.close_camera)
        settings.addWidget(self.close_button)
        self.photo_button = QPushButton("Abrir imagem")
        self.photo_button.clicked.connect(self.open_image)
        settings.addWidget(self.photo_button)
        self.return_button = QPushButton("Voltar à EMEET")
        self.return_button.clicked.connect(self.open_preferred_camera)
        settings.addWidget(self.return_button)

        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setTextFormat(Qt.TextFormat.PlainText)
        self.status.setObjectName("status")
        self.status.setMinimumHeight(66)
        settings.addWidget(self.status)
        self.settings_scroll.setWidget(settings_content)
        settings_container.addWidget(self.settings_scroll)
        sidebar_column.addWidget(self.settings_panel, 1)

        self.chart_panel = QFrame()
        self.chart_panel.setObjectName("diceChartPanel")
        self.chart_panel.setFixedWidth(280)
        self.chart_panel.setFixedHeight(440)
        chart_layout = QVBoxLayout(self.chart_panel)
        chart_layout.setContentsMargins(2, 2, 2, 2)
        self.dice_chart_panel = DiceChartPanel()
        self.dice_chart_panel.new_roll_button.clicked.connect(self.process_current_frame)
        self.dice_chart = self.dice_chart_panel.chart
        chart_layout.addWidget(self.dice_chart_panel)
        sidebar_column.addWidget(self.chart_panel)
        self.setStyleSheet("""
            QMainWindow { background: #080a0f; }
            QDialog { background: #121725; }
            QFrame#cameraPanel { background: #050608; border: 2px solid #ef4444; }
            QFrame#settingsPanel { background: #121725; border: 2px solid #6547ef; }
            QFrame#pipelinePanel { background: #0b1712; border: 2px solid #22c55e; }
            QFrame#diceChartPanel { background: #121725; border: 2px solid #facc15; }
            QLabel#diceChartTitle { color: #facc15; font-size: 16px; font-weight: 600; }
            QLabel#diceChartSummary { color: #e2e8f0; font-size: 13px; }
            QLabel#diceChartFormula { color: #94a3b8; font-size: 11px; }
            QTabWidget::pane { border: none; }
            QTabBar::tab { background: #1e293b; color: #cbd5e1; padding: 5px 12px; }
            QTabBar::tab:selected { background: #334155; color: #facc15; }
            QTableWidget { background: #121725; color: #e2e8f0; gridline-color: #334155; }
            QHeaderView::section { background: #263348; color: #cbd5e1; border: none; padding: 3px; }
            QScrollArea#pipelineStages, QWidget#stageViewport, QWidget#stagePreviews {
                background: #0b1712; border: none;
            }
            QFrame#stageCard { background: #0b1712; }
            QFrame#stageCard:hover { background: #132b20; }
            QScrollArea#analysisStages, QWidget#analysisViewport, QWidget#analysisPreviews {
                background: #121725; border: none;
            }
            QScrollArea#settingsScroll, QWidget#settingsViewport, QWidget#settingsContent {
                background: #121725; border: none;
            }
            QScrollBar:vertical { background: #1e293b; width: 10px; }
            QScrollBar::handle:vertical { background: #43516a; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
            QScrollBar:horizontal { background: #1e293b; height: 10px; }
            QScrollBar::handle:horizontal { background: #476256; min-width: 24px; }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: transparent;
            }
            QLabel { color: #cbd5e1; font-size: 14px; }
            QLabel#title { color: #f8fafc; font-size: 22px; font-weight: 600; }
            QLabel#status { color: #94a3b8; font-size: 12px; }
            QLabel#placeholder { background: #050608; }
            QLabel#pipelineTitle { color: #86efac; font-size: 15px; font-weight: 600; }
            QLabel#pipelineInfo { color: #a7b6ad; font-size: 12px; }
            QLabel#stagePreview { background: #050608; color: #94a3b8; font-size: 12px; }
            QLabel#stageCaption { color: #cbd5e1; font-size: 12px; }
            QCheckBox { color: #cbd5e1; font-size: 13px; }
            QPushButton, QComboBox, QDoubleSpinBox, QSpinBox {
                background: #263348; color: #f8fafc; border: 1px solid #43516a;
                border-radius: 6px; padding: 10px; font-size: 14px;
            }
            QComboBox QAbstractItemView {
                background: #263348; color: #f8fafc;
                selection-background-color: #2563eb;
            }
            QPushButton:hover { background: #354662; }
            QPushButton#primary { background: #2563eb; border-color: #2563eb; }
            QPushButton#primary:hover { background: #1d4ed8; }
            QPushButton:disabled { background: #1e293b; color: #64748b; }
            QComboBox:disabled { color: #64748b; }
            QSpinBox:disabled, QDoubleSpinBox:disabled { color: #64748b; background: #1e293b; }
            QSlider::groove:horizontal {
                height: 6px; background: #334155; border-radius: 3px;
            }
            QSlider::sub-page:horizontal { background: #6547ef; border-radius: 3px; }
            QSlider::handle:horizontal {
                background: #e2e8f0; width: 16px; margin: -5px 0; border-radius: 8px;
            }
            QSlider:disabled { background: transparent; }
            QTabWidget::pane { border: 1px solid #43516a; }
            QTabBar::tab { background: #263348; color: #f8fafc; padding: 10px; }
            QTabBar::tab:selected { background: #354662; }
            QListWidget, QTableWidget { background: #121725; color: #cbd5e1; }
            QHeaderView::section { background: #263348; color: #f8fafc; padding: 5px; }
            QHeaderView { background: #121725; }
        """)

        self.camera_combo.currentIndexChanged.connect(self.selection_changed)
        self.media_devices.videoInputsChanged.connect(self.refresh_devices)
        self.update_stage_captions()
        self.refresh_devices()

    def refresh_devices(self):
        """Atualiza os dispositivos preservando a seleção pelo identificador."""
        selected_id = self.camera_combo.currentData()
        self.devices = QMediaDevices.videoInputs()
        available_ids = [device.id() for device in self.devices]
        disconnected = (
            (self.camera is not None and self.camera.cameraDevice().id() not in available_ids)
            or (self.reopen_timer.isActive() and selected_id not in available_ids)
        )
        if disconnected:
            self.close_camera()

        self.camera_combo.blockSignals(True)
        self.camera_combo.clear()
        for device in self.devices:
            self.camera_combo.addItem(device.description(), device.id())
        if self.devices:
            default_index = self.preferred_camera_index()
            if default_index is None:
                default_index = next(
                    (index for index, device in enumerate(self.devices) if device.isDefault()),
                    0,
                )
            index = (
                available_ids.index(selected_id)
                if selected_id in available_ids
                else default_index
            )
            self.camera_combo.setCurrentIndex(index)
        else:
            self.camera_combo.addItem("Nenhuma webcam encontrada", None)
        self.camera_combo.blockSignals(False)
        self.camera_combo.setEnabled(bool(self.devices))
        self.refresh_quality(preserve=selected_id == self.camera_combo.currentData())
        self.update_buttons()

        if disconnected:
            self.status.setText("A webcam foi desconectada. Selecione outra câmera.")
        elif not self.devices:
            self.status.setText("Nenhuma webcam encontrada. Conecte uma câmera e atualize a lista.")
        elif self.camera is None and self.source != "photo":
            self.status.setText("Pronto para abrir a webcam selecionada.")

    def selection_changed(self):
        if self.camera is not None or self.reopen_timer.isActive():
            self.close_camera()
        self.refresh_quality()
        if self.source != "photo":
            self.status.setText("Clique em Abrir webcam para usar a câmera selecionada.")

    def preferred_camera_index(self):
        return next(
            (index for index, device in enumerate(self.devices)
             if DEFAULT_CAMERA_NAME.casefold() in device.description().casefold()),
            None,
        )

    def open_preferred_camera(self):
        """Abre a EMEET ao iniciar; informa sua ausência quando não está conectada."""
        self.refresh_devices()
        index = self.preferred_camera_index()
        if index is None:
            self.status.setText(
                f"{DEFAULT_CAMERA_NAME} não encontrada. Conecte a câmera e atualize a lista."
            )
            return
        self.camera_combo.setCurrentIndex(index)
        self.open_camera()

    def refresh_quality(self, preserve=False):
        previous = self.quality_combo.currentText() if preserve else DEFAULT_CAMERA_QUALITY
        selected_id = self.camera_combo.currentData()
        device = next((item for item in self.devices if item.id() == selected_id), None)
        self.quality_combo.blockSignals(True)
        self.quality_combo.clear()
        self.quality_combo.addItem("Automática", None)
        if device is not None:
            formats = {}
            for camera_format in device.videoFormats():
                size = camera_format.resolution()
                key = (size.width(), size.height(), camera_format.maxFrameRate())
                formats.setdefault(key, camera_format)
            for (width, height, fps), camera_format in sorted(
                formats.items(), key=lambda item: (item[0][0] * item[0][1], item[0][2]), reverse=True,
            ):
                label = f"{width} × {height} · {fps:g} fps"
                self.quality_combo.addItem(label, camera_format)
        index = self.quality_combo.findText(previous)
        self.quality_combo.setCurrentIndex(max(index, 0))
        self.quality_combo.blockSignals(False)
        self.quality_combo.setEnabled(device is not None and self.source != "photo")
        adjustments_enabled = device is not None or self.source == "photo"
        self.zoom_slider.setEnabled(adjustments_enabled)
        self.brightness_slider.setEnabled(adjustments_enabled)
        self.reset_button.setEnabled(adjustments_enabled)

    def quality_changed(self):
        if self.camera is not None or self.reopen_timer.isActive():
            self.close_camera()
            # Aguarda a liberação dos objetos da captura anterior pelo Qt.
            self.reopen_timer.start()
            self.status.setText("Aplicando a qualidade selecionada…")
            self.update_buttons()

    def zoom_changed(self, value):
        self.zoom_value.setText(f"{value / 100:.1f}×".replace(".", ","))
        self.video_widget.set_zoom(value)
        self.process_current_frame()

    def brightness_changed(self, value):
        self.brightness_value.setText(f"{value:+d}" if value else "0")
        self.video_widget.set_brightness(value)
        self.process_current_frame()

    def reset_adjustments(self):
        self.zoom_slider.setValue(100)
        self.brightness_slider.setValue(0)

    def update_buttons(self):
        busy = self.camera is not None or self.reopen_timer.isActive()
        self.open_button.setEnabled(bool(self.devices) and not busy)
        self.close_button.setEnabled(busy)
        self.return_button.setEnabled(self.source == "photo" and not busy)

    def toggle_pipeline(self, enabled):
        if enabled and (self.camera is not None or self.source == "photo"):
            if self.source == "camera":
                self.pipeline_timer.start()
            self.process_current_frame()
        else:
            self.pipeline_timer.stop()
            self.clear_pipeline()
            if not enabled:
                self.pipeline_info.setText("Processamento pausado.")

    def clear_pipeline(self):
        self.pipeline_result = None
        self.detection_result = None
        self.dice_chart_panel.set_dice_values([])
        self.dice_chart_panel.stop_observation()
        self.hovered_stage = None
        self.stage_images = []
        self.video_widget.set_stage_image()
        self.analysis_button.setEnabled(False)
        self.video_widget.detection = None
        self.video_widget.update()
        for preview in self.stage_previews:
            preview.clear()
            preview.setText("Aguardando imagem")
        self.pipeline_info.setText(
            "Abra a webcam para iniciar o processamento."
            if self.pipeline_enabled.isChecked() else "Processamento pausado."
        )

    def configure_candidate_filters(self):
        dialog = CandidateSettingsDialog(
            self.candidate_filters, self, self.preprocessing_parameters, self.pip_filters,
        )
        try:
            if dialog.exec() == QDialog.DialogCode.Accepted:
                self.apply_configuration(*dialog.selected_configuration())
        finally:
            dialog.deleteLater()

    def apply_configuration(self, parameters, candidate_filters, pip_filters):
        self.preprocessing_parameters = parameters
        self.candidate_filters = candidate_filters
        self.pip_filters = pip_filters
        self.update_stage_captions()
        self.process_current_frame()

    def update_stage_captions(self):
        p = self.preprocessing_parameters
        threshold_caption = (
            f"Threshold > {p.face_threshold}" if p.threshold_mode == "fixed"
            else "Otsu · automático" if p.threshold_mode == "otsu"
            else "Adaptativo · " + ("Gauss" if p.threshold_mode == "adaptive_gaussian" else "Média")
        )
        captions = [
            f"Colorida · {p.width} px", f"Cinza · {p.width} px",
            f"Gaussian Blur {p.blur_kernel}×{p.blur_kernel}", threshold_caption,
            f"Fechamento · {p.closing_iterations}×", f"Abertura · {p.opening_iterations}×",
            "Candidatos e dados", "Resultado final",
        ]
        for label, text in zip(self.stage_captions, captions):
            label.setText(text)
        self.binary_preview.setToolTip(
            f"Threshold fixo: pixels acima de {p.face_threshold}."
            if p.threshold_mode == "fixed" else
            "Threshold automático de Otsu." if p.threshold_mode == "otsu" else
            f"Threshold adaptativo; bloco {p.adaptive_block_size}×{p.adaptive_block_size}; C = {p.adaptive_c:g}."
        )
        self.closing_preview.setToolTip(
            f"Kernel elíptico {p.morphology_kernel}×{p.morphology_kernel}; fechamento {p.closing_iterations}×."
        )
        self.morphology_preview.setToolTip(
            f"Kernel elíptico {p.morphology_kernel}×{p.morphology_kernel}; abertura {p.opening_iterations}×."
            + (" Buracos internos preenchidos antes da abertura." if p.fill_face_holes else "")
        )

    def hover_stage(self, index, hovering):
        if hovering:
            self.hovered_stage = index
            self.update_hovered_stage()
        elif self.hovered_stage == index:
            self.hovered_stage = None
            self.video_widget.set_stage_image()

    def update_hovered_stage(self):
        index = self.hovered_stage
        if index is not None and 0 <= index < len(self.stage_images):
            self.video_widget.set_stage_image(
                self.stage_images[index], self.stage_captions[index].text(), binary=index in (3, 4, 5),
            )

    def show_face_analysis(self):
        if self.detection_result is not None:
            dialog = FaceAnalysisDialog(self.detection_result, self)
            try:
                dialog.exec()
            finally:
                dialog.deleteLater()

    def open_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Abrir fotografia de dados", "",
            "Imagens (*.jpg *.jpeg *.png *.bmp *.tif *.tiff);;Todos os arquivos (*)",
        )
        if path:
            self.load_image(path)

    def load_image(self, path):
        try:
            image = bgr_to_qimage(load_bgr_image(path))
        except ValueError as error:
            self.status.setText(str(error))
            return False
        self.close_camera()
        self.source = "photo"
        self._photo_fallback = False
        self.photo_path = str(path)
        self.photo_image = image
        self.video_widget.image = image.copy()
        self.video_widget.update()
        self.preview.setCurrentWidget(self.video_widget)
        self.refresh_quality(preserve=True)
        self.update_buttons()
        self.status.setText(f"Foto: {Path(path).name}")
        self.dice_chart_panel.start_new_roll()
        self.process_current_frame()
        return True

    def camera_frame_received(self):
        if self.camera is None or not self._awaiting_camera_frame:
            return
        self._awaiting_camera_frame = False
        self.start_timer.stop()
        if self.source == "photo":
            self.source = "camera"
            self.dice_chart_panel.start_new_roll()
            self._photo_fallback = False
            self.photo_image = QImage()
            self.photo_path = None
            self.clear_pipeline()
            self.refresh_quality(preserve=True)
            self.update_buttons()
        if self.pipeline_enabled.isChecked() and not self.pipeline_timer.isActive():
            self.pipeline_timer.start()
            self.process_current_frame()

    def process_current_frame(self):
        if not self.pipeline_enabled.isChecked() or not (self.video_widget.receiving or self.source == "photo"):
            return
        image = self.video_widget.adjusted_image()
        if image.isNull():
            return
        started = time.perf_counter()
        try:
            result = preprocess_frame(qimage_to_bgr(image), self.preprocessing_parameters)
            detection = detect_candidates(result, self.candidate_filters, self.pip_filters)
            images = [bgr_to_qimage(result.resized_bgr)]
            images.extend(gray_to_qimage(array) for array in (
                result.gray, result.blur, result.binary, result.closing, result.morphology,
            ))
            images.append(bgr_to_qimage(detection.annotated_bgr))
            images.append(bgr_to_qimage(detection.final_bgr))
        except (ValueError, RuntimeError, cv2.error) as error:
            self.pipeline_timer.stop()
            self.clear_pipeline()
            self.pipeline_info.setText(f"Falha no processamento: {error}")
            return
        self.pipeline_result = result
        self.detection_result = detection
        self.dice_chart_panel.set_dice_values(detection.dice_values)
        self.dice_chart_panel.observe_dice(detection.dice, time.monotonic(), photo=self.source == "photo")
        self.stage_images = images
        self.update_hovered_stage()
        self.analysis_button.setEnabled(True)
        self.video_widget.detection = detection
        self.video_widget.update()
        for preview, image in zip(self.stage_previews, images):
            # Nearest-neighbor mantém as máscaras apenas em preto/branco.
            transformation = (
                Qt.TransformationMode.FastTransformation
                if preview in (self.binary_preview, self.closing_preview, self.morphology_preview)
                else Qt.TransformationMode.SmoothTransformation
            )
            preview.setPixmap(QPixmap.fromImage(image).scaled(
                preview.size(), Qt.AspectRatioMode.KeepAspectRatio, transformation,
            ))
        height, width = result.gray.shape
        original_width, original_height = result.original_size
        elapsed_ms = (time.perf_counter() - started) * 1000
        self.pipeline_info.setText(
            f"{original_width} × {original_height} → {width} × {height} px · "
            f"{elapsed_ms:.1f} ms\n"
            f"Contornos: {detection.contour_count} · Candidatos: {len(detection.candidates)} · "
            f"Dados: {len(detection.dice)} · Soma total: {detection.total_sum}\n"
            + (" · ".join(f"Dado {die.die_id}: {len(die.pips)}" for die in detection.dice)
               or "Nenhum dado validado.")
        )

    def open_camera(self):
        if self.camera is not None:
            return
        selected_id = self.camera_combo.currentData()
        device = next((item for item in self.devices if item.id() == selected_id), None)
        if device is None:
            self.refresh_devices()
            return

        self._photo_fallback = self.source == "photo"
        self._awaiting_camera_frame = True
        self.pipeline_timer.stop()
        self.camera = QCamera(device, self)
        self.camera.activeChanged.connect(self.camera_active_changed)
        self.camera.errorOccurred.connect(self.camera_error)
        camera_format = self.quality_combo.currentData()
        if camera_format is not None:
            self.camera.setCameraFormat(camera_format)
        self.capture_session.setCamera(self.camera)
        self.video_widget.receiving = True
        self.status.setText(f"Abrindo {device.description()}…")
        self.update_buttons()
        self.start_timer.start()
        self.camera.start()

    def camera_active_changed(self, active):
        if self.sender() is not self.camera:
            return
        if active:
            self.preview.setCurrentWidget(self.video_widget)
            self.status.setText(f"Transmitindo: {self.camera.cameraDevice().description()}")
        else:
            self.close_camera()
            self.status.setText("A transmissão foi interrompida. Tente abrir a webcam novamente.")

    def camera_error(self, error, message):
        if self.sender() is not self.camera or error == QCamera.Error.NoError:
            return
        self.close_camera()
        self.status.setText(
            f"Não foi possível acessar a webcam: {message or 'erro de câmera'}. "
            "Verifique a permissão de câmera e se outro aplicativo está usando o dispositivo."
        )

    def camera_timed_out(self):
        self.close_camera()
        self.status.setText("A webcam não respondeu. Verifique a conexão e tente novamente.")

    def close_camera(self):
        """Libera a câmera e remove a última imagem da visualização."""
        self.start_timer.stop()
        self.startup_timer.stop()
        self.reopen_timer.stop()
        self.pipeline_timer.stop()
        self._awaiting_camera_frame = False
        restore_photo = self.source == "photo" or self._photo_fallback
        self.video_widget.clear()
        self.clear_pipeline()
        camera = self.camera
        self.camera = None
        if camera is not None:
            camera.activeChanged.disconnect(self.camera_active_changed)
            camera.errorOccurred.disconnect(self.camera_error)
            camera.stop()
            self.capture_session.setCamera(None)
            camera.deleteLater()
        if restore_photo and not self.photo_image.isNull():
            self.source = "photo"
            self._photo_fallback = False
            self.video_widget.image = self.photo_image.copy()
            self.video_widget.update()
            self.preview.setCurrentWidget(self.video_widget)
            self.status.setText(f"Foto: {Path(self.photo_path).name}")
            self.process_current_frame()
        else:
            self.source = "camera"
            self.preview.setCurrentWidget(self.placeholder)
            self.status.setText("Webcam fechada.")
        self.update_buttons()

    def closeEvent(self, event: QCloseEvent):
        self.source = "camera"
        self._photo_fallback = False
        self.photo_image = QImage()
        self.close_camera()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Projeto Dados")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = WebcamWindow()
    window.show()
    window.startup_timer.start(0)
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
