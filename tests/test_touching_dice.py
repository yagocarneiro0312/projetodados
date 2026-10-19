"""Pares de dados encostados, valores individuais e limites da separação."""

from dataclasses import replace
from pathlib import Path
import unittest

import cv2

from dice_detection import detect_candidates
from pipeline_parameters import emeet_calibration
from vision_pipeline import preprocess_frame
from test_dice_detection import draw_die, scene


class TouchingDiceTests(unittest.TestCase):
    def test_real_touching_dice_reproduce_failure_and_are_separated(self):
        parameters, faces, pips = emeet_calibration()
        paths = sorted((Path(__file__).parent / "fixtures" / "emeet_touching_13").glob("frame_*.png"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(frame=path.name):
                prepared = preprocess_frame(cv2.imread(str(path)), parameters)
                before = detect_candidates(prepared, replace(faces, separate_touching_dice=False), pips)
                self.assertEqual(before.dice_values, [])
                detection = detect_candidates(prepared, faces, pips)
                self.assertEqual(detection.dice_values, [1, 3])
                self.assertEqual(detection.total_sum, 4)
                self.assertEqual([die.die_id for die in detection.dice], [1, 2])

    def test_all_value_pairs_touching_horizontally_vertically_and_diagonally(self):
        parameters, faces, pips = emeet_calibration()
        for first in range(1, 7):
            for second in range(1, 7):
                image = scene()
                draw_die(image, 350, 250, first)
                draw_die(image, 450, 250, second)
                for angle in (0, 30, 45, 90):
                    with self.subTest(values=(first, second), angle=angle):
                        rotated = cv2.warpAffine(image, cv2.getRotationMatrix2D((450, 300), angle, 1),
                                                 (1000, 600), borderValue=(70,) * 3)
                        detection = detect_candidates(preprocess_frame(rotated, parameters), faces, pips)
                        self.assertEqual(sorted(detection.dice_values), sorted([first, second]))
                        self.assertEqual(detection.total_sum, first + second)

    def test_pair_with_a_blank_face_is_not_invented_as_two_dice(self):
        parameters, faces, pips = emeet_calibration()
        image = scene()
        draw_die(image, 350, 250, 0)
        draw_die(image, 450, 250, 3)
        detection = detect_candidates(preprocess_frame(image, parameters), faces, pips)
        self.assertEqual(detection.dice_values, [])

    def test_dark_dot_on_the_dividing_line_is_not_split_into_two_pips(self):
        parameters, faces, pips = emeet_calibration()
        image = scene()
        draw_die(image, 350, 250, 1)
        draw_die(image, 450, 250, 1)
        cv2.circle(image, (450, 300), 7, (20,) * 3, cv2.FILLED)
        detection = detect_candidates(preprocess_frame(image, parameters), faces, pips)
        self.assertEqual(detection.dice_values, [])
