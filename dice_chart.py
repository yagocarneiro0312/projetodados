"""Gráficos do frame atual e das últimas jogadas confirmadas."""

from PyQt6.QtCore import QRectF, Qt
from PyQt6.QtGui import QColor, QFont, QPainter, QPen
from PyQt6.QtWidgets import (
    QAbstractItemView, QHBoxLayout, QHeaderView, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget,
)

from dice_history import DiceHistory, StableRollTracker


class DiceChart(QWidget):
    LABELS = ("Dado 1", "Dado 2", "Total", "Divisão")
    COLORS = ("#60a5fa", "#a78bfa", "#34d399", "#facc15")

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values = (None, None, None, None)
        self.setAccessibleName("Gráfico dos dados em tempo real")
        self.setMinimumHeight(154)
        self.setMinimumWidth(240)
        self.update_accessibility()

    @staticmethod
    def format_value(value):
        return "—" if value is None else f"{value:g}".replace(".", ",")

    def set_dice_values(self, dice_values):
        first = dice_values[0] if dice_values else None
        second = dice_values[1] if len(dice_values) >= 2 else None
        total = first + second if first is not None and second is not None else None
        values = (first, second, total, total / 2 if total is not None else None)
        if values != self.values:
            self.values = values
            self.update_accessibility()
            self.update()

    def update_accessibility(self):
        self.setAccessibleDescription(" · ".join(
            f"{label}: {self.format_value(value)}"
            for label, value in zip(self.LABELS, self.values)
        ))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPointSize(9)
        painter.setFont(font)
        left, right, top, bottom = 24, self.width() - 5, 26, self.height() - 26
        plot_height = bottom - top
        # A soma varia de 2 a 12; uma escala fixa evita saltos entre frames.
        for tick in (0, 6, 12):
            y = bottom - tick / 12 * plot_height
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawLine(left, round(y), right, round(y))
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(QRectF(0, y - 8, left - 5, 16),
                             Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter, str(tick))
        slot_width = (right - left) / 4
        bar_width = min(36, slot_width - 14)
        for index, (label, value, color) in enumerate(zip(self.LABELS, self.values, self.COLORS)):
            center = left + slot_width * (index + .5)
            bar_height = 0 if value is None else value / 12 * plot_height
            y = bottom - bar_height
            if bar_height:
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor(color))
                painter.drawRoundedRect(QRectF(center - bar_width / 2, y, bar_width, bar_height), 4, 4)
            painter.setPen(QColor(color))
            value_font = QFont(font)
            value_font.setBold(True)
            value_font.setPointSize(11)
            painter.setFont(value_font)
            painter.drawText(QRectF(center - slot_width / 2, y - 25, slot_width, 22),
                             Qt.AlignmentFlag.AlignCenter, self.format_value(value))
            painter.setFont(font)
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(QRectF(center - slot_width / 2, bottom + 6, slot_width, 20),
                             Qt.AlignmentFlag.AlignCenter, label)


class FrequencyChart(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(18 * len(DiceHistory.MEAN_VALUES))
        self.setAccessibleName("Percentuais das médias por jogada, de 1 a 6")
        self.setToolTip(
            "Média da jogada = (Dado 1 + Dado 2) ÷ 2.\n"
            "Percentual = jogadas com essa média ÷ total de jogadas × 100."
        )
        self.set_counts((0,) * len(DiceHistory.MEAN_VALUES))

    @staticmethod
    def format_percent(count, total):
        value = f"{100 * count / total:.1f}" if total else "0"
        return value.removesuffix(".0").replace(".", ",") + "%"

    def set_counts(self, counts):
        self.counts = tuple(counts)
        total = sum(self.counts)
        self.setAccessibleDescription(" · ".join(
            f"Média {DiceChart.format_value(mean)}: {count} "
            f"{'jogada' if count == 1 else 'jogadas'} ({self.format_percent(count, total)})"
            for mean, count in zip(DiceHistory.MEAN_VALUES, self.counts)
        ))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont(self.font())
        font.setPointSize(8)
        painter.setFont(font)
        total, highest = sum(self.counts), max(self.counts)
        row_height = self.height() / len(DiceHistory.MEAN_VALUES)
        label_width = painter.fontMetrics().horizontalAdvance("Média 6,5") + 4
        value_width = painter.fontMetrics().horizontalAdvance("50 · 100%") + 4
        bar_left = label_width + 2
        bar_width = max(0, self.width() - bar_left - value_width - 6)
        for index, (mean, count) in enumerate(zip(DiceHistory.MEAN_VALUES, self.counts)):
            y = index * row_height
            painter.setPen(QColor("#e2e8f0"))
            painter.drawText(QRectF(0, y, label_width, row_height), Qt.AlignmentFlag.AlignVCenter,
                             f"Média {DiceChart.format_value(mean)}")
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#263348"))
            painter.drawRoundedRect(QRectF(bar_left, y + 4, bar_width, row_height - 8), 3, 3)
            if count:
                painter.setBrush(QColor("#facc15" if count == highest else "#60a5fa"))
                painter.drawRoundedRect(QRectF(bar_left, y + 4, bar_width * count / total, row_height - 8), 3, 3)
            painter.setPen(QColor("#e2e8f0"))
            percent = self.format_percent(count, total)
            painter.drawText(QRectF(bar_left + bar_width + 4, y, value_width, row_height),
                             Qt.AlignmentFlag.AlignVCenter, f"{count} · {percent}")


class DiceChartPanel(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.history = DiceHistory(limit=50)
        self.tracker = StableRollTracker()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs, 1)
        current = QWidget()
        self.tabs.addTab(current, "Agora")
        current_layout = QVBoxLayout(current)
        current_layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Dados em tempo real")
        title.setObjectName("diceChartTitle")
        current_layout.addWidget(title)
        self.chart = DiceChart()
        current_layout.addWidget(self.chart, 1)
        self.summary = QLabel("Aguardando dois dados")
        self.summary.setObjectName("diceChartSummary")
        self.summary.setWordWrap(True)
        current_layout.addWidget(self.summary)
        formula = QLabel("Divisão = (Dado 1 + Dado 2) ÷ 2")
        formula.setObjectName("diceChartFormula")
        formula.setAlignment(Qt.AlignmentFlag.AlignCenter)
        current_layout.addWidget(formula)
        history_page = QWidget()
        self.tabs.addTab(history_page, "Histórico")
        history_layout = QVBoxLayout(history_page)
        history_layout.setContentsMargins(4, 4, 4, 4)
        history_layout.setSpacing(4)
        self.history_summary = QLabel()
        self.history_summary.setWordWrap(True)
        self.history_summary.setStyleSheet("font-size: 11px;")
        history_layout.addWidget(self.history_summary)
        self.frequency_chart = FrequencyChart()
        history_layout.addWidget(self.frequency_chart)
        self.history_table = QTableWidget(0, 4)
        self.history_table.setAccessibleName("Últimas 50 jogadas, mais recente primeiro")
        self.history_table.setHorizontalHeaderLabels(["Jogada", "D1", "D2", "Média"])
        self.history_table.verticalHeader().hide()
        self.history_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.history_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.history_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.history_table.setMinimumHeight(64)
        self.history_table.setStyleSheet("font-size: 11px;")
        history_layout.addWidget(self.history_table, 1)
        buttons = QHBoxLayout()
        self.new_roll_button = QPushButton("Nova jogada")
        self.new_roll_button.setToolTip("Armar um novo registro, mesmo se os valores forem iguais aos anteriores.")
        self.new_roll_button.clicked.connect(self.start_new_roll)
        self.undo_button = QPushButton("Desfazer")
        self.undo_button.clicked.connect(self.undo_roll)
        self.clear_history_button = QPushButton("Limpar")
        self.clear_history_button.clicked.connect(self.clear_history)
        for button in (self.new_roll_button, self.undo_button, self.clear_history_button):
            button.setStyleSheet("padding: 4px; font-size: 11px;")
            buttons.addWidget(button)
        history_layout.addLayout(buttons)
        self.recording_status = QLabel()
        self.recording_status.setWordWrap(True)
        self.recording_status.setStyleSheet("font-size: 10px; color: #94a3b8;")
        self.recording_status.setToolTip(
            "Dois dados estáveis por 1 s registram uma jogada. Retire ambos por 0,8 s "
            "ou clique em Nova jogada antes de lançar novamente. Fotos registram uma vez ao abrir. "
            "O histórico fica disponível durante esta sessão."
        )
        layout.addWidget(self.recording_status)
        self.refresh_history()
        self.update_recording_status()
        self.tabs.setCurrentIndex(1)

    def observe_dice(self, dice, now, photo=False):
        values = [len(die.pips) for die in dice]
        boxes = [die.bbox for die in dice]
        confirmed = self.tracker.observe(values, boxes, now, immediate=photo)
        if confirmed is not None:
            self.history.record(confirmed)
            self.refresh_history()
        self.update_recording_status()

    def stop_observation(self):
        self.tracker.cancel_pending()
        self.update_recording_status()

    def start_new_roll(self):
        self.tracker.new_roll()
        self.update_recording_status()

    def undo_roll(self):
        if self.history.rolls:
            self.history.rolls.pop()
            self.refresh_history()

    def clear_history(self):
        self.history.clear()
        self.refresh_history()

    def update_recording_status(self):
        self.recording_status.setText(
            "Automático: aguardando 2 dados estáveis (1 s)." if self.tracker.armed else
            "Registrado. Retire os dados ou use Nova jogada."
        )

    def refresh_history(self):
        history = self.history
        count = len(history.rolls)
        mean = "—" if history.mean is None else f"{history.mean:.2f}".replace(".", ",")
        leaders = "; ".join(map(self.chart.format_value, history.most_frequent)) or "—"
        self.history_summary.setText(
            f"Últimas {count}/50 jogadas · Média geral: {mean}\n"
            f"Médias mais frequentes: {leaders}"
        )
        self.frequency_chart.set_counts(history.frequencies)
        self.history_table.setRowCount(count)
        for row, roll in enumerate(reversed(history.rolls)):
            values = (roll.number, *roll.values, self.chart.format_value(roll.mean))
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.history_table.setItem(row, column, item)
        self.history_table.scrollToTop()
        self.undo_button.setEnabled(bool(count))
        self.clear_history_button.setEnabled(bool(count))

    def set_dice_values(self, dice_values):
        self.chart.set_dice_values(dice_values)
        if len(dice_values) < 2:
            self.summary.setText("Aguardando dois dados")
        else:
            first, second, total, division = self.chart.values
            self.summary.setText(
                f"{first} + {second} = {total}   ·   {total} ÷ 2 = {self.chart.format_value(division)}"
            )
