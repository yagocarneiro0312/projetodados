"""Exportação e geração pela ferramenta externa, sem abrir o detector."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtWidgets import QApplication

from dice_simulation_dialog import DiceSimulationDialog


class SimulationInterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_external_window_generates_and_exports_the_displayed_sample(self):
        dialog = DiceSimulationDialog()
        self.addCleanup(dialog.close)
        dialog.show()
        self.assertIsNone(dialog.parent())
        self.assertTrue(dialog.isVisible())
        self.assertEqual(len(dialog.sample.history.rolls), 300)
        self.assertIn("300 jogadas simuladas", dialog.summary.text())
        dialog.count_combo.setCurrentIndex(0)
        dialog.seed_input.setText("12345")
        dialog.generate_button.click()
        self.assertEqual(len(dialog.sample.history.rolls), 200)
        self.assertEqual(dialog.table.rowCount(), 200)
        self.assertEqual(sum(dialog.sample.history.frequencies), 200)
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / "grafico.png"
            csv_path = Path(directory) / "jogadas.csv"
            with patch("dice_simulation_dialog.QFileDialog.getSaveFileName", return_value=(str(image_path), "")):
                dialog.save_button.click()
            with patch("dice_simulation_dialog.QFileDialog.getSaveFileName", return_value=(str(csv_path), "")):
                dialog.csv_button.click()
            self.assertEqual(image_path.read_bytes()[:8], b"\x89PNG\r\n\x1a\n")
            self.assertEqual(len(csv_path.read_text(encoding="utf-8-sig").splitlines()), 201)
        dialog.seed_input.setText("2147483648")
        with patch("dice_simulation_dialog.QMessageBox.warning") as warning:
            dialog.generate_button.click()
        warning.assert_called_once()
        self.assertEqual(len(dialog.sample.history.rolls), 200)


if __name__ == "__main__":
    unittest.main()
