"""Regressões de rotação, escala, posição, contraste e perspectiva para 1–6."""

from dataclasses import replace
import unittest

import cv2
import numpy as np

from dice_detection import detect_candidates
from pipeline_parameters import emeet_calibration
from synthetic_dice import render_scene
from vision_pipeline import preprocess_frame


class RotationRobustnessTests(unittest.TestCase):
    def detect(self, image, legacy=False):
        parameters, faces, pips = emeet_calibration()
        if legacy:
            parameters = replace(parameters, fill_face_holes=False)
            pips = replace(pips, threshold_mode="adaptive_gaussian", normalize_scale=False)
        return detect_candidates(preprocess_frame(image, parameters), faces, pips)

    def test_reproduces_missing_face_and_merged_pips_before_the_fix(self):
        cases = [
            (dict(side=30, pip_radius=.10, angle=0), []),
            (dict(side=36, pip_radius=.11, angle=30), [2]),
        ]
        for settings, previous in cases:
            with self.subTest(settings=settings):
                image = render_scene(value=6, **settings)
                self.assertEqual(self.detect(image, legacy=True).dice_values, previous)
                self.assertEqual(self.detect(image).dice_values, [6])

    def test_six_survives_a_full_turn_at_fifteen_degree_intervals(self):
        for angle in range(0, 360, 15):
            with self.subTest(angle=angle):
                image = render_scene(value=6, side=36, angle=angle, pip_radius=.11)
                self.assertEqual(self.detect(image).dice_values, [6])

    def test_all_faces_across_rotations_sizes_and_perspective(self):
        conditions = [
            dict(side=30, pip_radius=.10),
            dict(side=36, pip_radius=.11),
            dict(side=36, pip_radius=.095, perspective=.6, blur=1),
            dict(side=42, pip_radius=.082, perspective=1, blur=.8),
        ]
        for settings in conditions:
            for value in range(1, 7):
                for angle in range(0, 180, 15):
                    with self.subTest(value=value, angle=angle, settings=settings):
                        image = render_scene(value=value, angle=angle, **settings)
                        self.assertEqual(self.detect(image).dice_values, [value])

    def test_position_low_contrast_and_noise_do_not_choose_the_value(self):
        for value in range(1, 7):
            for center, angle in [((130, 110), 0), ((810, 120), 35),
                                  ((170, 430), 90), ((800, 420), 145)]:
                with self.subTest(value=value, center=center, angle=angle):
                    image = render_scene(value=value, center=center, angle=angle, side=46,
                                         background=165, face=205, noise=1.2, perspective=.4)
                    self.assertEqual(self.detect(image).dice_values, [value])

    def test_blank_seven_and_elongated_marks_do_not_become_dice(self):
        for value in (0, 7):
            for angle in (0, 30, 60, 90):
                with self.subTest(value=value, angle=angle):
                    self.assertEqual(self.detect(render_scene(value=value, angle=angle)).dice_values, [])
        image = render_scene(value=0, side=56)
        cv2.line(image, (484, 280), (516, 280), (20,) * 3, 3)
        self.assertEqual(self.detect(image).dice_values, [])

    def test_seeded_variations_outside_the_regular_angle_grid(self):
        rng = np.random.default_rng(45619)
        for index in range(72):
            value = index % 6 + 1
            settings = dict(
                side=float(rng.uniform(34, 68)),
                center=(float(rng.uniform(90, 910)), float(rng.uniform(90, 470))),
                angle=float(rng.uniform(0, 360)), pip_radius=float(rng.uniform(.07, .10)),
                perspective=float(rng.uniform(0, .5)), blur=float(rng.uniform(.3, .8)),
                face=int(rng.integers(205, 246)), noise=.8,
            )
            with self.subTest(index=index, value=value, settings=settings):
                self.assertEqual(self.detect(render_scene(value=value, **settings)).dice_values, [value])

    def test_normalized_measurements_return_original_coordinates_and_masks(self):
        image = render_scene(value=6, angle=35, center=(220, 180), side=36)
        detection = self.detect(image)
        self.assertEqual(detection.dice_values, [6])
        die = detection.dice[0]
        x, y, width, height = die.bbox
        self.assertEqual(die.dark_mask.shape, (height, width))
        self.assertEqual(die.cleaned_mask.shape, (height, width))
        self.assertFalse(np.any(die.dark_mask[die.face_mask == 0]))
        for pip in die.pips:
            self.assertTrue(x <= pip.center[0] < x + width)
            self.assertTrue(y <= pip.center[1] < y + height)
            self.assertLess(pip.radius, 8)
        radians = np.radians(35)
        rotation = np.array([[np.cos(radians), -np.sin(radians)], [np.sin(radians), np.cos(radians)]])
        expected = [np.array((220, 180)) + rotation @ ((np.array((px, py)) - .5) * 36)
                    for px in (.25, .75) for py in (.25, .5, .75)]
        for center in expected:
            self.assertLess(min(np.linalg.norm(np.array(pip.center) - center) for pip in die.pips), 1.2)


if __name__ == "__main__":
    unittest.main()
