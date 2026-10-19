"""Validação de configuração e aplicação dos parâmetros ao pré-processamento."""

import unittest

import cv2
import numpy as np

from pipeline_parameters import CandidateFilters, PipFilters, PreprocessingParameters
from vision_pipeline import preprocess_frame


class ParameterTests(unittest.TestCase):
    def test_invalid_preparation_and_pip_parameters(self):
        cases = [
            (PreprocessingParameters, {"width": 0}),
            (PreprocessingParameters, {"width": float("inf")}),
            (PreprocessingParameters, {"blur_kernel": 4}),
            (PreprocessingParameters, {"morphology_kernel": 0}),
            (PreprocessingParameters, {"face_threshold": 256}),
            (PreprocessingParameters, {"closing_iterations": -1}),
            (PipFilters, {"threshold": -1}), (PipFilters, {"opening_kernel": 2}),
            (PipFilters, {"min_circularity": float("nan")}),
            (PipFilters, {"min_relative_area": .2, "max_relative_area": .1}),
            (PipFilters, {"max_relative_area": float("inf")}),
            (CandidateFilters, {"min_aspect_ratio": 1, "max_aspect_ratio": 1}),
            (PreprocessingParameters, {"threshold_mode": "unknown"}),
            (PreprocessingParameters, {"adaptive_block_size": 2}),
            (PipFilters, {"adaptive_block_size": 1}),
            (PipFilters, {"adaptive_c": float("inf")}),
            (PreprocessingParameters, {"fill_face_holes": 1}),
            (PipFilters, {"normalize_scale": "sim"}),
        ]
        for cls, values in cases:
            with self.subTest(cls=cls, values=values):
                with self.assertRaises(ValueError):
                    cls(**values)

    def test_linear_resize_and_custom_width_threshold_and_zero_iterations(self):
        rng = np.random.default_rng(123)
        image = rng.integers(0, 256, (300, 1800, 3), dtype=np.uint8)
        parameters = PreprocessingParameters(width=500, blur_kernel=1, face_threshold=100,
                                              closing_iterations=0, opening_iterations=0)
        result = preprocess_frame(image, parameters)
        expected = cv2.resize(image, None, fx=500/1800, fy=500/1800, interpolation=cv2.INTER_LINEAR)
        np.testing.assert_array_equal(result.resized_bgr, expected)
        np.testing.assert_array_equal(result.binary, np.where(result.gray > 100, 255, 0))
        np.testing.assert_array_equal(result.closing, result.binary)
        np.testing.assert_array_equal(result.morphology, result.binary)

    def test_adaptive_faces_recover_die_in_dim_region_of_gradient(self):
        from dice_detection import detect_candidates

        row = np.linspace(50, 150, 1000, dtype=np.uint8)
        image = np.repeat(np.repeat(row[None, :, None], 600, axis=0), 3, axis=2)
        for x, positions in [(120, [(40, 40)]), (750, [(20, 20), (60, 20), (20, 60), (60, 60)])]:
            cv2.rectangle(image, (x, 200), (x + 80, 280), (int(row[x]) + 70,) * 3, cv2.FILLED)
            for dx, dy in positions:
                cv2.circle(image, (x + dx, 200 + dy), 6, (20,) * 3, cv2.FILLED)
        self.assertEqual(detect_candidates(preprocess_frame(image)).dice_values, [4])
        for mode in ("adaptive_gaussian", "adaptive_mean"):
            with self.subTest(mode=mode):
                prepared = preprocess_frame(image, PreprocessingParameters(threshold_mode=mode))
                self.assertEqual(detect_candidates(prepared).dice_values, [1, 4])
                self.assertEqual(prepared.binary.dtype, np.uint8)
                self.assertTrue(set(np.unique(prepared.binary)).issubset({0, 255}))

    def test_adaptive_points_recover_pips_brighter_than_fixed_cutoff(self):
        from dice_detection import detect_candidates

        image = np.full((600, 1000, 3), 70, dtype=np.uint8)
        cv2.rectangle(image, (200, 200), (300, 300), (245,) * 3, cv2.FILLED)
        for x in (225, 275):
            for y in (225, 250, 275):
                cv2.circle(image, (x, y), 7, (160,) * 3, cv2.FILLED)
        prepared = preprocess_frame(image)
        self.assertEqual(detect_candidates(prepared).dice_values, [])
        for mode in ("adaptive_gaussian", "adaptive_mean"):
            with self.subTest(mode=mode):
                detection = detect_candidates(prepared, pip_filters=PipFilters(threshold_mode=mode))
                self.assertEqual(detection.dice_values, [6])
                candidate = detection.dice[0]
                self.assertFalse(np.any(candidate.dark_mask[candidate.face_mask == 0]))

    def test_large_adaptive_block_handles_small_input(self):
        prepared = preprocess_frame(np.full((2, 3, 3), 120, dtype=np.uint8),
                                    PreprocessingParameters(width=3, threshold_mode="adaptive_gaussian"))
        self.assertEqual(prepared.binary.shape, (2, 3))
        self.assertFalse(np.any(prepared.binary))

    def test_local_otsu_ignores_pixels_outside_the_face(self):
        from thresholding import threshold_image

        mask = np.zeros((80, 80), np.uint8)
        mask[20:60, 20:60] = 255
        outputs = []
        for background in (0, 130, 255):
            gray = np.full_like(mask, background)
            gray[mask > 0] = 230
            cv2.circle(gray, (40, 40), 5, 160, -1)
            dark = threshold_image(gray, "otsu", 0, 31, 7, inverted=True, mask=mask)
            self.assertEqual(dark[40, 40], 255)
            self.assertEqual(dark[25, 25], 0)
            outputs.append(dark[mask > 0])
        np.testing.assert_array_equal(outputs[0], outputs[1])
        np.testing.assert_array_equal(outputs[1], outputs[2])
