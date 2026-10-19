"""Ferramenta externa de simulação; não abre o detector nem acessa a câmera."""

import sys

from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import QApplication

from dice_simulation_dialog import DiceSimulationDialog


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Simulador Estatístico de Dados")
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    window = DiceSimulationDialog()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
