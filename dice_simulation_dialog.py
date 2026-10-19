"""Janela de demonstração estatística com dados identificados como simulados."""

from pathlib import Path

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIntValidator
from PyQt6.QtWidgets import (
    QAbstractItemView, QComboBox, QDialog, QFileDialog, QHBoxLayout,
    QHeaderView, QLabel, QLineEdit, QMessageBox, QPushButton,
    QTableWidget, QTableWidgetItem, QVBoxLayout,
)

from dice_simulation import create_simulation_figure, generate_simulation


class DiceSimulationDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Simulador Estatístico de Dados")
        self.resize(1080, 820)
        self.setMinimumSize(900, 700)
        self.setStyleSheet("""
            QDialog { background: #121725; }
            QLabel { color: #cbd5e1; font-size: 14px; }
            QPushButton, QComboBox {
                background: #263348; color: #f8fafc; border: 1px solid #43516a;
                border-radius: 6px; padding: 10px; font-size: 14px;
            }
            QPushButton:hover { background: #354662; }
            QComboBox QAbstractItemView {
                background: #263348; color: #f8fafc; selection-background-color: #2563eb;
            }
            QTableWidget { background: #121725; color: #cbd5e1; gridline-color: #334155; }
            QHeaderView::section { background: #263348; color: #f8fafc; padding: 5px; }
            QHeaderView { background: #121725; }
            QScrollBar:vertical { background: #1e293b; width: 10px; }
            QScrollBar::handle:vertical { background: #43516a; min-height: 24px; }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(10)
        title = QLabel("Simulação de dois dados")
        title.setStyleSheet("font-size: 21px; font-weight: 600; color: #facc15;")
        layout.addWidget(title)
        explanation = QLabel(
            "Dados simulados: dois dados justos e independentes por jogada. "
            "A amostra varia naturalmente a cada geração."
        )
        explanation.setWordWrap(True)
        explanation.setStyleSheet("font-size: 12px;")
        layout.addWidget(explanation)

        controls = QHBoxLayout()
        self.count_combo = QComboBox()
        for count in (200, 300):
            self.count_combo.addItem(f"{count} jogadas", count)
        self.count_combo.setCurrentIndex(1)
        self.count_combo.setAccessibleName("Quantidade de jogadas simuladas")
        controls.addWidget(self.count_combo)
        self.seed_input = QLineEdit()
        self.seed_input.setPlaceholderText("Semente automática")
        self.seed_input.setAccessibleName("Semente opcional para reproduzir a simulação")
        self.seed_input.setToolTip("Informe a mesma semente e quantidade para reproduzir os resultados.")
        self.seed_input.setValidator(QIntValidator(0, 2147483647, self))
        self.seed_input.setStyleSheet(
            "background: #263348; color: #f8fafc; border: 1px solid #43516a; "
            "border-radius: 6px; padding: 10px; font-size: 13px;"
        )
        controls.addWidget(self.seed_input, 1)
        self.generate_button = QPushButton("Gerar amostra")
        self.generate_button.clicked.connect(self.generate_sample)
        controls.addWidget(self.generate_button)
        self.save_button = QPushButton("Salvar gráfico")
        self.save_button.clicked.connect(self.save_figure)
        controls.addWidget(self.save_button)
        self.csv_button = QPushButton("Exportar CSV")
        self.csv_button.clicked.connect(self.export_csv)
        controls.addWidget(self.csv_button)
        layout.addLayout(controls)

        self.summary = QLabel()
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size: 12px;")
        layout.addWidget(self.summary)
        self.canvas = None
        self.figure_layout = QVBoxLayout()
        self.figure_layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(self.figure_layout, 1)
        self.table = QTableWidget(0, 4)
        self.table.setAccessibleName("Todas as jogadas simuladas, mais recente primeiro")
        self.table.setHorizontalHeaderLabels(["Jogada simulada", "Dado 1", "Dado 2", "Média"])
        self.table.verticalHeader().hide()
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setMinimumHeight(100)
        self.table.setMaximumHeight(135)
        layout.addWidget(self.table)
        self.generate_sample()

    def generate_sample(self):
        seed_text = self.seed_input.text().strip()
        if seed_text and not self.seed_input.hasAcceptableInput():
            QMessageBox.warning(self, "Semente inválida", "Informe um inteiro entre 0 e 2147483647.")
            return
        self.sample = generate_simulation(self.count_combo.currentData(), int(seed_text) if seed_text else None)
        count = len(self.sample.history.rolls)
        mean = f"{self.sample.history.mean:.3f}".replace(".", ",")
        self.summary.setText(
            f"{count} jogadas simuladas · {2 * count} resultados de dados · "
            f"Média geral: {mean} · Semente: {self.sample.seed}"
        )
        previous = self.canvas
        self.canvas = FigureCanvasQTAgg(create_simulation_figure(self.sample))
        self.canvas.setAccessibleName("Distribuição das médias simuladas e curva teórica triangular")
        if previous is not None:
            self.figure_layout.removeWidget(previous)
            previous.setParent(None)
            previous.deleteLater()
        self.figure_layout.addWidget(self.canvas)
        self.table.setRowCount(count)
        for row, roll in enumerate(reversed(self.sample.history.rolls)):
            for column, value in enumerate((roll.number, *roll.values, f"{roll.mean:g}".replace(".", ","))):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, column, item)
        self.table.scrollToTop()
        self.canvas.draw_idle()

    def save_figure(self):
        count = len(self.sample.history.rolls)
        path, _ = QFileDialog.getSaveFileName(
            self, "Salvar gráfico da simulação", f"simulacao_{count}_jogadas.png", "Imagem PNG (*.png)"
        )
        if path:
            try:
                self.canvas.figure.savefig(str(Path(path).with_suffix(".png")), dpi=300, facecolor="white")
            except OSError as error:
                QMessageBox.warning(self, "Não foi possível salvar o gráfico", str(error))

    def export_csv(self):
        count = len(self.sample.history.rolls)
        path, _ = QFileDialog.getSaveFileName(
            self, "Exportar jogadas simuladas", f"simulacao_{count}_jogadas.csv", "CSV (*.csv)"
        )
        if path:
            try:
                self.sample.export_csv(Path(path).with_suffix(".csv"))
            except OSError as error:
                QMessageBox.warning(self, "Não foi possível exportar as jogadas", str(error))
