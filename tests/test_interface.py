"""Fluxo de fotos, configuração e troca de fontes sem exigir câmera física."""

import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import cv2
from PyQt6.QtCore import QEvent, QObject, QPoint, QSize, Qt, pyqtSignal
from PyQt6.QtTest import QTest
from PyQt6.QtMultimedia import QCamera, QVideoFrame
from PyQt6.QtWidgets import QApplication, QDialog

import main
from candidate_settings import CandidateSettingsDialog
from face_analysis import FaceAnalysisDialog
from pipeline_parameters import CandidateFilters, PipFilters, PreprocessingParameters, c920_calibration, emeet_calibration
from qt_image_utils import bgr_to_qimage, qimage_to_bgr
from test_dice_detection import draw_die, scene


FIXTURES = Path(__file__).parent / "fixtures"


class FakeDevice:
    def id(self):
        return b"external-emeet"

    def description(self):
        return "EMEET SmartCam S600"

    def isDefault(self):
        return False

    def videoFormats(self):
        return []


class FakeCamera(QObject):
    activeChanged = pyqtSignal(bool)
    errorOccurred = pyqtSignal(object, str)

    def __init__(self, device, parent):
        super().__init__(parent)
        self.device = device
        self.active = False
        self.camera_format = None

    def setCameraFormat(self, camera_format):
        self.camera_format = camera_format

    def cameraDevice(self):
        return self.device

    def start(self):
        self.active = True
        self.activeChanged.emit(True)

    def stop(self):
        self.active = False


class InterfaceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.inputs_patch = patch.object(main.QMediaDevices, "videoInputs", return_value=[])
        self.inputs = self.inputs_patch.start()
        self.addCleanup(self.inputs_patch.stop)
        self.window = main.WebcamWindow()
        self.addCleanup(self.window.close)
        self.window.show()
        self.app.processEvents()

    def load_reference(self):
        self.assertTrue(self.window.load_image(FIXTURES / "pdf_two_dice.png"))
        self.assertEqual(self.window.detection_result.dice_values, [4, 6])

    def install_fake_camera(self):
        self.inputs.return_value = [FakeDevice()]
        self.window.capture_session = Mock()
        camera_patch = patch.object(main, "QCamera", side_effect=FakeCamera)
        camera_patch.start()
        self.addCleanup(camera_patch.stop)

    def test_startup_uses_adaptive_faces_and_robust_points_without_configuration(self):
        window = self.window
        self.assertEqual(window.preprocessing_parameters.threshold_mode, "adaptive_gaussian")
        self.assertEqual(window.pip_filters.threshold_mode, "otsu")
        self.assertTrue(window.preprocessing_parameters.fill_face_holes)
        self.assertTrue(window.pip_filters.normalize_scale)
        self.assertIn("Adaptativo", window.stage_captions[3].text())
        self.load_reference()
        p = window.preprocessing_parameters
        expected = cv2.adaptiveThreshold(window.pipeline_result.blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, p.adaptive_block_size, p.adaptive_c)
        np.testing.assert_array_equal(window.pipeline_result.binary, expected)
        for candidate in window.detection_result.candidates:
            self.assertEqual(candidate.dark_mask.shape, candidate.roi_gray.shape)
            self.assertFalse(np.any(candidate.dark_mask[candidate.face_mask == 0]))

    def test_camera_starts_with_robust_profile_without_opening_filters(self):
        self.install_fake_camera()
        window = self.window
        window.open_preferred_camera()
        window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(np.full((480, 640, 3), 70, dtype=np.uint8))))
        self.assertEqual(window.source, "camera")
        self.assertTrue(window.pipeline_timer.isActive())
        self.assertEqual(window.preprocessing_parameters.threshold_mode, "adaptive_gaussian")
        self.assertEqual(window.pip_filters.threshold_mode, "otsu")
        self.assertTrue(window.pip_filters.normalize_scale)
        self.assertIn("Adaptativo", window.stage_captions[3].text())

    def test_startup_prefers_emeet_and_applies_720p_60fps(self):
        self.install_fake_camera()
        formats = []
        for width, height, fps in [(1920, 1080, 30), (1280, 720, 120), (1280, 720, 60), (1280, 720, 30)]:
            camera_format = Mock()
            camera_format.resolution.return_value = QSize(width, height)
            camera_format.maxFrameRate.return_value = fps
            formats.append(camera_format)
        preferred = Mock(wraps=FakeDevice())
        preferred.videoFormats.return_value = formats
        other = Mock(wraps=FakeDevice())
        other.id.return_value = b"external-c920"
        other.description.return_value = "HD Pro Webcam C920"
        other.isDefault.return_value = True
        self.inputs.return_value = [other, preferred]
        self.window.refresh_devices()
        self.assertEqual(self.window.camera_combo.currentData(), preferred.id())
        self.assertEqual(self.window.quality_combo.currentText(), "1280 × 720 · 60 fps")
        self.window.startup_timer.start(0)
        self.app.processEvents()
        self.assertEqual(self.window.camera.cameraDevice().id(), preferred.id())
        self.assertIs(self.window.camera.camera_format, formats[2])
        self.assertTrue(self.window.camera.active)

    def test_refresh_preserves_manual_quality_and_missing_default_uses_automatic(self):
        camera_format = Mock()
        camera_format.resolution.return_value = QSize(1920, 1080)
        camera_format.maxFrameRate.return_value = 30
        device = Mock(wraps=FakeDevice())
        device.videoFormats.return_value = [camera_format]
        self.inputs.return_value = [device]
        self.window.refresh_devices()
        self.assertEqual(self.window.quality_combo.currentText(), "Automática")
        self.window.quality_combo.setCurrentIndex(1)
        self.window.refresh_devices()
        self.assertIs(self.window.quality_combo.currentData(), camera_format)

    def test_unicode_photo_pipeline_pause_adjustments_and_resume(self):
        with tempfile.TemporaryDirectory(prefix="dados-") as directory:
            path = Path(directory) / "fotografia çã.png"
            shutil.copyfile(FIXTURES / "pdf_two_dice.png", path)
            self.assertTrue(self.window.load_image(path))
        window = self.window
        self.assertEqual(window.source, "photo")
        self.assertIsNone(window.camera)
        self.assertFalse(window.pipeline_timer.isActive())
        self.assertTrue(window.zoom_slider.isEnabled())
        self.assertFalse(window.quality_combo.isEnabled())
        self.assertEqual(len(window.stage_previews), 8)
        self.assertTrue(all(not label.pixmap().isNull() for label in window.stage_previews))
        self.assertIn("Soma total: 10", window.pipeline_info.text())
        self.assertIn("Dado 1: 4", window.pipeline_info.text())
        original = qimage_to_bgr(window.photo_image)
        window.pipeline_enabled.setChecked(False)
        self.assertIsNone(window.detection_result)
        self.assertFalse(window.analysis_button.isEnabled())
        window.zoom_slider.setValue(200)
        window.brightness_slider.setValue(10)
        window.pipeline_enabled.setChecked(True)
        self.assertIsNotNone(window.pipeline_result)
        self.assertLess(window.pipeline_result.original_size[0], original.shape[1])
        np.testing.assert_array_equal(qimage_to_bgr(window.photo_image), original)
        window.reset_adjustments()
        self.assertEqual(window.detection_result.dice_values, [4, 6])
        self.assertFalse(window.pipeline_timer.isActive())

    def test_cancel_and_invalid_file_preserve_source_and_results(self):
        self.load_reference()
        before = self.window.detection_result
        with patch.object(main.QFileDialog, "getOpenFileName", return_value=("", "")):
            self.window.open_image()
        self.assertIs(self.window.detection_result, before)
        with tempfile.TemporaryDirectory() as directory:
            invalid = Path(directory) / "invalid.jpg"
            invalid.write_bytes(b"invalid image")
            self.assertFalse(self.window.load_image(invalid))
        self.assertIs(self.window.detection_result, before)
        self.assertEqual(self.window.source, "photo")
        self.assertFalse(self.window.load_image("missing-image.jpg"))
        self.assertIs(self.window.detection_result, before)

    def test_absent_emeet_keeps_photo_and_never_opens_another_camera(self):
        self.load_reference()
        before = self.window.detection_result
        other = Mock(wraps=FakeDevice())
        other.description.return_value = "HD Pro Webcam C920"
        self.inputs.return_value = [other]
        with patch.object(main, "QCamera") as camera:
            self.window.open_preferred_camera()
        camera.assert_not_called()
        self.assertEqual(self.window.source, "photo")
        self.assertIs(self.window.detection_result, before)
        self.assertIn("EMEET SmartCam S600 não encontrada", self.window.status.text())

    def test_return_to_camera_switches_after_first_valid_frame(self):
        self.load_reference()
        self.install_fake_camera()
        self.window.open_preferred_camera()
        self.assertEqual(self.window.source, "photo")
        self.assertTrue(self.window._photo_fallback)
        frame = QVideoFrame(bgr_to_qimage(np.full((480, 640, 3), 70, dtype=np.uint8)))
        self.window.video_widget.receive_frame(frame)
        self.assertEqual(self.window.source, "camera")
        self.assertFalse(self.window._photo_fallback)
        self.assertTrue(self.window.pipeline_timer.isActive())
        self.assertEqual(self.window.pipeline_result.original_size, (640, 480))
        self.assertEqual(self.window.detection_result.dice_values, [])
        self.assertTrue(self.window.quality_combo.isEnabled())
        self.window.close_camera()
        self.assertIsNone(self.window.pipeline_result)
        self.assertFalse(self.window.video_widget.receiving)

    def test_camera_failure_and_timeout_restore_photo(self):
        for failure in ("error", "timeout"):
            with self.subTest(failure=failure):
                self.load_reference()
                if failure == "error":
                    self.install_fake_camera()
                self.window.open_preferred_camera()
                if failure == "error":
                    self.window.camera.errorOccurred.emit(QCamera.Error.CameraError, "camera occupied")
                else:
                    self.window.camera_timed_out()
                self.assertIsNone(self.window.camera)
                self.assertEqual(self.window.source, "photo")
                self.assertEqual(self.window.detection_result.dice_values, [4, 6])
                self.assertTrue(self.window.return_button.isEnabled())

    def test_disconnection_clears_camera_results(self):
        self.install_fake_camera()
        self.window.open_preferred_camera()
        self.window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(np.zeros((480, 640, 3), dtype=np.uint8))))
        self.inputs.return_value = []
        self.window.refresh_devices()
        self.assertIsNone(self.window.camera)
        self.assertIsNone(self.window.detection_result)
        self.assertIn("desconectada", self.window.status.text())

    def test_parameter_apply_cancel_defaults_and_invalid_kernel(self):
        self.load_reference()
        dialog = CandidateSettingsDialog(self.window.candidate_filters, self.window)
        self.addCleanup(dialog.close)
        dialog.preparation_fields["blur_kernel"].setValue(4)
        dialog.accept()
        self.assertNotEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.assertIn("ímpar", dialog.error_label.text())
        dialog.preparation_fields["blur_kernel"].setValue(3)
        dialog.preparation_fields["width"].setValue(800)
        dialog.pip_fields["threshold"].setValue(120)
        dialog.accept()
        self.assertEqual(dialog.result(), QDialog.DialogCode.Accepted)
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertEqual(self.window.pipeline_result.gray.shape[1], 800)
        self.assertEqual(self.window.stage_captions[0].text(), "Colorida · 800 px")
        self.assertEqual(self.window.stage_captions[2].text(), "Gaussian Blur 3×3")
        self.assertEqual(self.window.pip_filters.threshold, 120)
        cancelled = CandidateSettingsDialog(self.window.candidate_filters, self.window,
                                             self.window.preprocessing_parameters, self.window.pip_filters)
        self.addCleanup(cancelled.close)
        cancelled.preparation_fields["width"].setValue(600)
        with patch.object(main, "CandidateSettingsDialog", return_value=cancelled), \
                patch.object(cancelled, "exec", return_value=QDialog.DialogCode.Rejected):
            self.window.configure_candidate_filters()
        self.assertEqual(self.window.preprocessing_parameters.width, 800)
        dialog.restore_defaults()
        self.assertEqual(dialog.selected_configuration(), (PreprocessingParameters(), CandidateFilters(), PipFilters()))
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertEqual(self.window.detection_result.dice_values, [4, 6])

    def test_inspector_has_five_views_metrics_and_stable_snapshot(self):
        self.load_reference()
        # Insere uma face sem pontos para exercitar a análise de rejeições;
        # o perfil calibrado já elimina os candidatos de fundo da fotografia.
        image = qimage_to_bgr(self.window.video_widget.image)
        cv2.rectangle(image, (20, 20), (70, 70), (245,) * 3, cv2.FILLED)
        self.window.video_widget.image = bgr_to_qimage(image)
        self.window.process_current_frame()
        dialog = FaceAnalysisDialog(self.window.detection_result, self.window)
        self.addCleanup(dialog.close)
        self.assertEqual(len(dialog.image_labels), 5)
        self.assertTrue(all(not label.pixmap().isNull() for label in dialog.image_labels))
        self.assertEqual(dialog.table.rowCount(), len(dialog.detection.candidates[0].measurements))
        snapshot = dialog.detection.candidates[0].roi_bgr.copy()
        self.window.detection_result.candidates[0].roi_bgr[:] = 0
        self.window.load_image(FIXTURES / "photo_six.png")
        self.assertEqual(dialog.detection.dice_values, [4, 6])
        np.testing.assert_array_equal(dialog.detection.candidates[0].roi_bgr, snapshot)
        rejected_index = next(i for i, candidate in enumerate(dialog.detection.candidates) if not candidate.is_die)
        dialog.candidate_list.setCurrentRow(rejected_index)
        self.assertIn("Rejeitado", dialog.summary.text())
        self.assertIn("Nenhum ponto válido", dialog.summary.text())
        accepted_index = next(i for i, candidate in enumerate(dialog.detection.candidates) if candidate.is_die)
        dialog.candidate_list.setCurrentRow(accepted_index)
        self.assertIn("Dado 1: 4", dialog.summary.text())
        self.assertIn("Solidez", dialog.summary.text())

    def test_narrow_layout_preserves_camera_settings_and_pipeline(self):
        self.load_reference()
        self.window.resize(900, 920)
        self.app.processEvents()
        self.assertEqual(self.window.pipeline_panel.width(), self.window.camera_panel.width())
        self.assertGreater(self.window.settings_panel.x(), self.window.camera_panel.x())
        self.assertGreater(self.window.pipeline_panel.y(), self.window.camera_panel.y())
        self.assertGreater(self.window.pipeline_scroll.horizontalScrollBar().maximum(), 0)
        self.assertEqual(self.window.chart_panel.x(), self.window.settings_panel.x())
        self.assertGreater(self.window.pipeline_panel.y(), self.window.chart_panel.y())
        self.assertEqual(self.window.pipeline_panel.geometry().bottom(), self.window.chart_panel.geometry().bottom())
        self.assertEqual(self.window.camera_panel.y(), self.window.settings_panel.y())
        self.assertGreater(self.window.camera_panel.height(), self.window.settings_panel.height())
        self.assertEqual(self.window.chart_panel.width(), self.window.settings_panel.width())
        self.assertGreaterEqual(self.window.dice_chart.width(), 240)

    def test_live_chart_updates_from_camera_timer_and_clears_when_paused(self):
        self.install_fake_camera()
        window = self.window
        window.open_preferred_camera()
        for values in ((3, 3), (5, 4)):
            with self.subTest(values=values):
                image = scene()
                for index, value in enumerate(values):
                    draw_die(image, 200 + index * 250, 200, value)
                window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(image)))
                QTest.qWait(150)
                self.app.processEvents()
                total = sum(values)
                self.assertEqual(window.dice_chart.values, (*values, total, total / 2))
        self.assertIn("4,5", window.dice_chart.accessibleDescription())
        window.pipeline_enabled.setChecked(False)
        self.assertEqual(window.dice_chart.values, (None,) * 4)
        window.pipeline_enabled.setChecked(True)
        self.assertEqual(window.dice_chart.values, (5, 4, 9, 4.5))
        window.close_camera()
        self.assertEqual(window.dice_chart.values, (None,) * 4)

    def test_chart_requires_two_dice_and_updates_when_changing_photos(self):
        self.load_reference()
        self.assertEqual(self.window.dice_chart.values, (4, 6, 10, 5))
        self.assertTrue(self.window.load_image(FIXTURES / "photo_six.png"))
        self.assertEqual(self.window.dice_chart.values, (6, None, None, None))
        self.assertEqual(self.window.dice_chart_panel.summary.text(), "Aguardando dois dados")
        self.load_reference()
        self.assertEqual(self.window.dice_chart.values, (4, 6, 10, 5))

    def test_touching_dice_update_the_chart_with_individual_values(self):
        self.assertTrue(self.window.load_image(FIXTURES / "emeet_touching_13" / "frame_03.png"))
        self.assertEqual(self.window.detection_result.dice_values, [1, 3])
        self.assertEqual(self.window.dice_chart.values, (1, 3, 4, 2))

    def test_photo_history_counts_once_and_keeps_statistics_when_pausing(self):
        self.load_reference()
        panel = self.window.dice_chart_panel
        self.assertEqual(panel.history.frequencies, (0,) * 8 + (1, 0, 0))
        self.assertEqual(panel.frequency_chart.counts, panel.history.frequencies)
        self.assertIn("Média 5: 1 jogada (100%)", panel.frequency_chart.accessibleDescription())
        for _ in range(3):
            self.window.process_current_frame()
        self.window.pipeline_enabled.setChecked(False)
        self.window.pipeline_enabled.setChecked(True)
        self.assertEqual(len(panel.history.rolls), 1)
        self.assertEqual(panel.history_table.item(0, 1).text(), "4")
        self.assertEqual(panel.history_table.item(0, 2).text(), "6")
        self.assertIn("5,00", panel.history_summary.text())
        self.assertTrue(self.window.load_image(FIXTURES / "emeet_touching_13" / "frame_03.png"))
        self.assertEqual(len(panel.history.rolls), 2)
        self.assertEqual(panel.history_table.item(0, 1).text(), "1")
        self.assertEqual(panel.history.mean, 3.5)
        self.assertIn("Média 2: 1 jogada (50%)", panel.frequency_chart.accessibleDescription())
        self.assertIn("Média 5: 1 jogada (50%)", panel.frequency_chart.accessibleDescription())
        panel.undo_button.click()
        self.assertEqual(panel.history.mean, 5)
        self.assertIn("Média 2: 0 jogadas (0%)", panel.frequency_chart.accessibleDescription())
        self.assertIn("Média 5: 1 jogada (100%)", panel.frequency_chart.accessibleDescription())
        self.assertFalse(panel.tracker.armed)
        panel.clear_history_button.click()
        self.assertEqual(panel.history_table.rowCount(), 0)
        self.assertEqual(panel.frequency_chart.counts, (0,) * 11)
        self.window.process_current_frame()
        self.assertEqual(len(panel.history.rolls), 0)
        panel.new_roll_button.click()
        self.assertEqual(len(panel.history.rolls), 1)
        self.assertEqual(panel.history.rolls[0].values, (1, 3))

    def test_history_percentages_group_pairs_with_the_same_fractional_mean(self):
        panel = self.window.dice_chart_panel
        for values in ([2, 5], [3, 4], [1, 2]):
            panel.history.record(values)
        panel.refresh_history()
        description = panel.frequency_chart.accessibleDescription()
        self.assertIn("Média 3,5: 2 jogadas (66,7%)", description)
        self.assertIn("Média 1,5: 1 jogada (33,3%)", description)
        self.assertNotIn("Face", description)
        self.assertIn("Médias mais frequentes: 3,5", panel.history_summary.text())
        self.assertEqual(panel.history_table.item(0, 3).text(), "1,5")
        self.assertEqual(panel.history_table.item(1, 3).text(), "3,5")

    def test_camera_history_records_one_roll_until_manually_rearmed(self):
        self.install_fake_camera()
        window = self.window
        panel = window.dice_chart_panel
        panel.tracker.stable_seconds = 0
        window.open_preferred_camera()
        image = scene()
        draw_die(image, 200, 200, 3)
        draw_die(image, 450, 200, 6)
        window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(image)))
        window.process_current_frame()
        self.assertEqual(len(panel.history.rolls), 1)
        window.process_current_frame()
        window.pipeline_enabled.setChecked(False)
        window.pipeline_enabled.setChecked(True)
        self.assertEqual(len(panel.history.rolls), 1)
        panel.new_roll_button.click()
        self.assertEqual(len(panel.history.rolls), 2)

    def test_hover_over_caption_and_thumbnail_shows_stage_then_restores_source(self):
        self.load_reference()
        window = self.window
        original = qimage_to_bgr(window.video_widget.image)
        QTest.mouseMove(window.status, QPoint(5, 5))
        self.app.processEvents()
        QTest.mouseMove(window.stage_cards[1].caption, QPoint(10, 5))
        self.app.processEvents()
        self.assertEqual(window.hovered_stage, 1)
        self.assertEqual(window.video_widget.stage_title, "Cinza · 1000 px")
        stage = window.video_widget.stage_image
        widget = window.video_widget
        scale = min(widget.width() / stage.width(), widget.height() / stage.height())
        left = int(np.ceil((widget.width() - stage.width() * scale) / 2)) + 2
        right = int(np.floor((widget.width() + stage.width() * scale) / 2)) - 2
        top = max(40, int(np.ceil((widget.height() - stage.height() * scale) / 2)) + 2)
        bottom = int(np.floor((widget.height() + stage.height() * scale) / 2)) - 2
        preview = qimage_to_bgr(widget.grab().toImage())[top:bottom, left:right]
        np.testing.assert_array_equal(preview[:, :, 0], preview[:, :, 1])
        np.testing.assert_array_equal(preview[:, :, 1], preview[:, :, 2])
        QTest.mouseMove(window.gray_preview, QPoint(20, 20))
        self.app.processEvents()
        self.assertEqual(window.hovered_stage, 1)
        np.testing.assert_array_equal(qimage_to_bgr(window.video_widget.image), original)
        QTest.mouseMove(window.status, QPoint(5, 5))
        self.app.processEvents()
        self.assertIsNone(window.hovered_stage)
        self.assertTrue(window.video_widget.stage_image.isNull())
        self.assertEqual(window.detection_result.dice_values, [4, 6])

    def test_hover_stays_current_and_does_not_change_processing_source(self):
        self.load_reference()
        window = self.window
        for index, card in enumerate(window.stage_cards):
            with self.subTest(stage=index):
                self.app.sendEvent(card, QEvent(QEvent.Type.Enter))
                self.assertEqual(window.hovered_stage, index)
                np.testing.assert_array_equal(qimage_to_bgr(window.video_widget.stage_image),
                                              qimage_to_bgr(window.stage_images[index]))
                self.assertEqual(window.video_widget.stage_binary, index in (3, 4, 5))
                self.app.sendEvent(card, QEvent(QEvent.Type.Leave))
                self.assertTrue(window.video_widget.stage_image.isNull())
        window.hover_stage(3, True)
        window.brightness_slider.setValue(10)
        np.testing.assert_array_equal(qimage_to_bgr(window.video_widget.stage_image),
                                      qimage_to_bgr(window.stage_images[3]))
        window.hover_stage(1, True)
        window.hover_stage(3, False)  # Uma saída atrasada não remove a nova etapa.
        self.assertEqual(window.hovered_stage, 1)
        window.reset_adjustments()
        self.assertEqual(window.detection_result.dice_values, [4, 6])
        window.pipeline_enabled.setChecked(False)
        self.assertTrue(window.video_widget.stage_image.isNull())
        self.assertEqual(window.stage_images, [])
        self.assertIsNone(window.hovered_stage)

    def test_live_hover_receives_updated_frames_and_clears_on_close(self):
        self.install_fake_camera()
        window = self.window
        window.open_preferred_camera()
        window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(np.full((480, 640, 3), 70, dtype=np.uint8))))
        window.hover_stage(1, True)
        before = qimage_to_bgr(window.video_widget.stage_image)
        window.video_widget.receive_frame(QVideoFrame(bgr_to_qimage(np.full((480, 640, 3), 100, dtype=np.uint8))))
        window.process_current_frame()
        after = qimage_to_bgr(window.video_widget.stage_image)
        self.assertFalse(np.array_equal(before, after))
        self.assertTrue(window.pipeline_timer.isActive())
        self.assertEqual(window.video_widget.adjusted_image().pixelColor(0, 0).red(), 100)
        window.close_camera()
        self.assertTrue(window.video_widget.stage_image.isNull())
        self.assertEqual(window.stage_images, [])

    def test_adaptive_dialog_controls_apply_and_restore_pdf_defaults(self):
        self.load_reference()
        dialog = CandidateSettingsDialog(self.window.candidate_filters, self.window)
        self.addCleanup(dialog.close)
        for group in (dialog.preparation_fields, dialog.pip_fields):
            self.assertFalse(group["adaptive_block_size"].isEnabled())
            group["threshold_mode"].setCurrentIndex(group["threshold_mode"].findData("adaptive_gaussian"))
            self.assertTrue(group["adaptive_block_size"].isEnabled())
            self.assertTrue(group["adaptive_c"].isEnabled())
            self.assertFalse(group["face_threshold" if "face_threshold" in group else "threshold"].isEnabled())
        dialog.preparation_fields["adaptive_block_size"].setValue(100)
        dialog.accept()
        self.assertIn("ímpar", dialog.error_label.text())
        dialog.preparation_fields["adaptive_block_size"].setValue(101)
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertEqual(self.window.preprocessing_parameters.threshold_mode, "adaptive_gaussian")
        self.assertEqual(self.window.pip_filters.threshold_mode, "adaptive_gaussian")
        self.assertIn("Adaptativo", self.window.stage_captions[3].text())
        self.assertIn("C = -10", self.window.binary_preview.toolTip())
        self.assertEqual(self.window.detection_result.dice_values, [4, 6])
        dialog.restore_defaults()
        self.assertFalse(dialog.preparation_fields["adaptive_c"].isEnabled())
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertEqual(self.window.stage_captions[3].text(), "Threshold > 190")

    def test_c920_profile_button_restores_calibrated_configuration(self):
        expected = c920_calibration()
        dialog = CandidateSettingsDialog(CandidateFilters(), self.window)
        self.addCleanup(dialog.close)
        dialog.restore_calibration()
        self.assertEqual(dialog.selected_configuration(), expected)
        self.assertTrue(dialog.preparation_fields["adaptive_c"].isEnabled())
        self.assertEqual(dialog.fields["min_solidity"].value(), .85)
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertTrue(self.window.load_image(FIXTURES / "c920_456" / "frame_00.png"))
        self.assertEqual(sorted(self.window.detection_result.dice_values), [4, 5, 6])
        self.assertEqual(self.window.detection_result.total_sum, 15)

    def test_otsu_controls_and_robust_options_can_be_selected_and_restored(self):
        dialog = CandidateSettingsDialog(self.window.candidate_filters, self.window,
                                         self.window.preprocessing_parameters, self.window.pip_filters)
        self.addCleanup(dialog.close)
        self.assertTrue(dialog.preparation_fields["fill_face_holes"].isChecked())
        self.assertTrue(dialog.pip_fields["normalize_scale"].isChecked())
        for group in (dialog.preparation_fields, dialog.pip_fields):
            group["threshold_mode"].setCurrentIndex(group["threshold_mode"].findData("otsu"))
            self.assertFalse(group["adaptive_block_size"].isEnabled())
            self.assertFalse(group["adaptive_c"].isEnabled())
            self.assertFalse(group["face_threshold" if "face_threshold" in group else "threshold"].isEnabled())
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertIn("Otsu", self.window.stage_captions[3].text())
        dialog.restore_defaults()
        self.assertFalse(dialog.preparation_fields["fill_face_holes"].isChecked())
        self.assertFalse(dialog.pip_fields["normalize_scale"].isChecked())

    def test_emeet_profile_is_default_and_can_be_restored(self):
        expected = emeet_calibration()
        self.assertEqual((self.window.preprocessing_parameters, self.window.candidate_filters,
                          self.window.pip_filters), expected)
        dialog = CandidateSettingsDialog(CandidateFilters(), self.window)
        self.addCleanup(dialog.close)
        dialog.restore_emeet_calibration()
        self.assertEqual(dialog.selected_configuration(), expected)
        self.window.apply_configuration(*dialog.selected_configuration())
        self.assertIn("3×3", self.window.stage_captions[2].text())
        self.assertIn("0×", self.window.stage_captions[4].text())
        self.assertTrue(self.window.load_image(FIXTURES / "emeet_54" / "frame_06.png"))
        self.assertEqual(self.window.detection_result.dice_values, [5, 4])
        self.assertEqual(self.window.detection_result.total_sum, 9)


if __name__ == "__main__":
    unittest.main()
