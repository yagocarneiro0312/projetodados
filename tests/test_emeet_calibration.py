"""Regressão de dois dados pequenos ao lado de traços claros do mousepad."""

from pathlib import Path
import unittest

import cv2
import numpy as np

from dice_detection import detect_candidates
from pipeline_parameters import c920_calibration, emeet_calibration
from vision_pipeline import preprocess_frame
from test_dice_detection import draw_die, scene


class EmeetCalibrationTests(unittest.TestCase):
    def test_real_emeet_frames_detect_both_dice_without_background_false_positives(self):
        paths = sorted((Path(__file__).parent / "fixtures" / "emeet_54").glob("frame_*.png"))
        self.assertEqual(len(paths), 7)
        old_parameters, old_faces, old_pips = c920_calibration()
        parameters, faces, pips = emeet_calibration()
        old_failures = 0
        for path in paths:
            with self.subTest(frame=path.name):
                image = cv2.imread(str(path))
                old = detect_candidates(preprocess_frame(image, old_parameters), old_faces, old_pips)
                old_failures += old.dice_values != [5, 4]
                detection = detect_candidates(preprocess_frame(image, parameters), faces, pips)
                self.assertEqual(detection.dice_values, [5, 4])
                self.assertEqual(detection.total_sum, 9)
        self.assertGreater(old_failures, 0)

    def test_faces_are_not_merged_with_nearby_background_lines(self):
        image = np.full((562, 1000, 3), 30, dtype=np.uint8)
        for x, y, value in [(450, 235, 5), (510, 220, 4)]:
            cv2.rectangle(image, (x, y), (x + 34, y + 34), (245,) * 3, cv2.FILLED)
            points = [(9, 9), (25, 9), (9, 25), (25, 25)]
            if value == 5:
                points.append((17, 17))
            for dx, dy in points:
                cv2.circle(image, (x + dx, y + dy), 3, (15,) * 3, cv2.FILLED)
        cv2.line(image, (552, 210), (575, 210), (160,) * 3, 3)
        cv2.line(image, (552, 210), (552, 270), (160,) * 3, 3)
        parameters, faces, pips = c920_calibration()
        self.assertEqual(detect_candidates(preprocess_frame(image, parameters), faces, pips).dice_values, [5])
        parameters, faces, pips = emeet_calibration()
        self.assertEqual(detect_candidates(preprocess_frame(image, parameters), faces, pips).dice_values, [5, 4])

    def test_profile_accepts_all_six_values_and_rotated_faces(self):
        parameters, faces, pips = emeet_calibration()
        for value in range(1, 7):
            for angle in (0, 30, 45):
                with self.subTest(value=value, angle=angle):
                    image = scene()
                    draw_die(image, 350, 250, value)
                    image = cv2.warpAffine(image, cv2.getRotationMatrix2D((400, 300), angle, 1),
                                          (1000, 600), borderValue=(70,) * 3)
                    self.assertEqual(detect_candidates(preprocess_frame(image, parameters), faces, pips).dice_values,
                                     [value])
