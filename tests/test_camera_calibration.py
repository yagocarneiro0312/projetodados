"""Regressão do perfil C920 com faces reais e contornos irregulares de fundo."""

from pathlib import Path
from dataclasses import replace
import unittest

import cv2

from dice_detection import detect_candidates
from pipeline_parameters import CandidateFilters, PipFilters, PreprocessingParameters, c920_calibration
from vision_pipeline import preprocess_frame
from test_dice_detection import draw_die, scene


class CalibrationTests(unittest.TestCase):
    def test_real_c920_frames_have_three_dice_and_sum_fifteen(self):
        parameters, filters, pip_filters = c920_calibration()
        paths = sorted((Path(__file__).parent / "fixtures" / "c920_456").glob("frame_*.png"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(frame=path.name):
                result = detect_candidates(preprocess_frame(cv2.imread(str(path)), parameters), filters, pip_filters)
                self.assertEqual(sorted(result.dice_values), [4, 5, 6])
                self.assertEqual(result.total_sum, 15)
                self.assertEqual(len(result.candidates), 3)
                self.assertTrue(all(candidate.solidity > .85 for candidate in result.dice))
                # Os IDs são por posição; a ordem física desta cena é 6, 4, 5.
                self.assertEqual(result.dice_values, [6, 4, 5])

    def test_baseline_reproduces_false_positive_and_calibration_removes_it(self):
        image = cv2.imread(str(Path(__file__).parent / "fixtures" / "c920_456" / "frame_00.png"))
        baseline = detect_candidates(preprocess_frame(image, PreprocessingParameters(threshold_mode="adaptive_gaussian")),
                                     pip_filters=PipFilters(threshold_mode="adaptive_gaussian"))
        self.assertGreater(len(baseline.dice), 3)
        parameters, filters, pip_filters = c920_calibration()
        calibrated = detect_candidates(preprocess_frame(image, parameters), filters, pip_filters)
        self.assertEqual(sorted(calibrated.dice_values), [4, 5, 6])

    def test_solidity_rejects_concave_region_with_a_valid_dark_dot(self):
        image = scene()
        image[220:380, 475:525] = 245
        image[275:325, 420:580] = 245
        cv2.circle(image, (500, 300), 8, (20,) * 3, cv2.FILLED)
        prepared = preprocess_frame(image)
        unfiltered = detect_candidates(prepared)
        self.assertEqual(unfiltered.dice_values, [1])
        self.assertLess(unfiltered.dice[0].solidity, .85)
        filtered = detect_candidates(prepared, CandidateFilters(min_solidity=.85))
        self.assertEqual(filtered.candidates, [])

    def test_convex_mask_recovers_fourth_pip_lost_at_face_boundary(self):
        image = cv2.imread(str(Path(__file__).parent / "fixtures" / "c920_456" / "frame_33.png"))
        parameters, filters, pips = c920_calibration()
        prepared = preprocess_frame(image, parameters)
        raw_mask = detect_candidates(prepared, replace(filters, convex_face_mask=False), pips)
        convex_mask = detect_candidates(prepared, filters, pips)
        self.assertEqual(sorted(raw_mask.dice_values), [3, 5, 6])
        self.assertEqual(sorted(convex_mask.dice_values), [4, 5, 6])
        self.assertEqual(len(convex_mask.candidates), 3)
        self.assertGreater(cv2.countNonZero(convex_mask.dice[1].face_mask),
                           cv2.countNonZero(raw_mask.dice[1].face_mask))

    def test_calibration_still_accepts_one_through_six_and_rotated_faces(self):
        parameters, filters, pip_filters = c920_calibration()
        for value in range(1, 7):
            for angle in (0, 30, 45):
                with self.subTest(value=value, angle=angle):
                    image = scene()
                    draw_die(image, 350, 250, value)
                    if angle:
                        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((400, 300), angle, 1),
                                              (1000, 600), borderValue=(70,) * 3)
                    result = detect_candidates(preprocess_frame(image, parameters), filters, pip_filters)
                    self.assertEqual(result.dice_values, [value])
