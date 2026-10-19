"""Candidatos geométricos e contagem de pontos em cenas sintéticas conhecidas."""

import unittest
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from dice_detection import CandidateFilters, PipFilters, detect_candidates
from vision_pipeline import preprocess_frame


PATTERNS = {
    1: [(50, 50)],
    2: [(25, 25), (75, 75)],
    3: [(25, 25), (50, 50), (75, 75)],
    4: [(25, 25), (75, 25), (25, 75), (75, 75)],
    5: [(25, 25), (75, 25), (50, 50), (25, 75), (75, 75)],
    6: [(25, 25), (75, 25), (25, 50), (75, 50), (25, 75), (75, 75)],
    7: [(25, 25), (75, 25), (25, 50), (50, 50), (75, 50), (25, 75), (75, 75)],
}


def draw_die(image, x, y, value):
    cv2.rectangle(image, (x, y), (x + 100, y + 100), (245, 245, 245), cv2.FILLED)
    for dx, dy in PATTERNS.get(value, []):
        cv2.circle(image, (x + dx, y + dy), 7, (20, 20, 20), cv2.FILLED)


def scene():
    return np.full((600, 1000, 3), 70, dtype=np.uint8)


class CandidateTests(unittest.TestCase):
    def test_detects_values_one_through_six_in_same_scene(self):
        image = scene()
        for value in range(1, 7):
            draw_die(image, 60 + (value - 1) * 145, 220, value)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.candidates), 6)
        self.assertEqual([len(die.pips) for die in result.dice], [1, 2, 3, 4, 5, 6])
        self.assertEqual(sum(len(die.pips) for die in result.dice), 21)
        self.assertTrue(all(candidate.face_bgr.size for candidate in result.candidates))

    def test_white_reflection_without_pips_is_not_a_die(self):
        image = scene()
        draw_die(image, 200, 200, 0)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.candidates[0].pips), 0)
        self.assertEqual(result.dice, [])

    def test_seven_valid_pips_reject_candidate(self):
        image = scene()
        draw_die(image, 200, 200, 7)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.candidates[0].pips), 7)
        self.assertEqual(result.dice, [])

    def test_external_contours_do_not_count_pip_holes_as_faces(self):
        image = scene()
        draw_die(image, 200, 200, 6)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(result.contour_count, 1)
        self.assertEqual(len(result.candidates), 1)

    def test_filters_remove_tiny_large_elongated_and_low_extent_regions(self):
        image = scene()
        draw_die(image, 50, 50, 1)
        image[50:62, 300:312] = 245
        image[100:120, 400:580] = 245
        image[210:570, 620:980] = 245
        image[300:410, 100:114] = 245
        image[396:410, 100:210] = 245
        result = detect_candidates(preprocess_frame(image))
        self.assertGreaterEqual(result.contour_count, 5)
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.dice), 1)

    def test_custom_area_filter_removes_otherwise_valid_face(self):
        image = scene()
        draw_die(image, 200, 200, 1)
        result = detect_candidates(preprocess_frame(image), CandidateFilters(min_relative_area=0.03, max_relative_area=0.05))
        self.assertEqual(len(result.candidates), 0)

    def test_rotated_face_is_accepted_and_outside_dark_regions_are_ignored(self):
        image = scene()
        draw_die(image, 350, 250, 4)
        matrix = cv2.getRotationMatrix2D((400, 300), 30, 1)
        image = cv2.warpAffine(image, matrix, (1000, 600), borderValue=(70, 70, 70))
        cv2.circle(image, (500, 300), 12, (0, 0, 0), cv2.FILLED)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.dice), 1)
        self.assertEqual(len(result.dice[0].pips), 4)
        self.assertLess(result.dice[0].extent, 0.7)

    def test_dark_elongated_mark_is_not_a_valid_pip(self):
        image = scene()
        draw_die(image, 200, 200, 0)
        image[245:252, 220:280] = 20
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.candidates[0].pips), 0)
        self.assertEqual(result.dice, [])

    def test_boundary_mark_excluded_by_face_mask_is_not_a_pip(self):
        image = scene()
        draw_die(image, 200, 200, 0)
        cv2.circle(image, (201, 250), 8, (20, 20, 20), cv2.FILLED)
        result = detect_candidates(preprocess_frame(image))
        self.assertEqual(len(result.candidates), 1)
        self.assertEqual(len(result.candidates[0].pips), 0)
        self.assertEqual(result.dice, [])
        candidate = result.candidates[0]
        self.assertFalse(np.any(candidate.dark_mask[candidate.face_mask == 0]))

    def test_reference_images_reproduce_pdf_values(self):
        for filename, expected in [("pdf_two_dice.png", [4, 6]), ("photo_six.png", [6])]:
            with self.subTest(filename=filename):
                image = cv2.imread(str(Path(__file__).parent / "fixtures" / filename))
                result = detect_candidates(preprocess_frame(image))
                self.assertEqual(result.dice_values, expected)
                self.assertEqual(result.total_sum, sum(expected))
                self.assertEqual([die.die_id for die in result.dice], list(range(1, len(expected) + 1)))

    def test_face_and_point_filter_boundaries_are_strict(self):
        image = scene()
        draw_die(image, 200, 200, 1)
        prepared = preprocess_frame(image)
        candidate = detect_candidates(prepared).candidates[0]
        for name, value in [
            ("min_relative_area", candidate.relative_area),
            ("max_relative_area", candidate.relative_area),
            ("min_aspect_ratio", candidate.aspect_ratio),
            ("max_aspect_ratio", candidate.aspect_ratio), ("min_extent", candidate.extent),
            ("min_solidity", candidate.solidity),
            ("min_rectangularity", candidate.rectangularity),
        ]:
            with self.subTest(face_boundary=name):
                self.assertEqual(detect_candidates(prepared, replace(CandidateFilters(), **{name: value})).candidates, [])
        point = candidate.measurements[0]
        for name, value in [
            ("min_relative_area", point.relative_area), ("max_relative_area", point.relative_area),
            ("min_aspect_ratio", point.aspect_ratio), ("max_aspect_ratio", point.aspect_ratio),
            ("min_circularity", point.circularity),
        ]:
            with self.subTest(point_boundary=name):
                result = detect_candidates(prepared, pip_filters=replace(PipFilters(), **{name: value}))
                self.assertEqual(result.dice_values, [])
                self.assertFalse(result.candidates[0].measurements[0].accepted)

    def test_pip_threshold_includes_130_and_excludes_131(self):
        for intensity, expected in [(130, [1]), (131, [])]:
            with self.subTest(intensity=intensity):
                image = scene()
                draw_die(image, 200, 200, 0)
                cv2.circle(image, (250, 250), 7, (intensity,) * 3, cv2.FILLED)
                result = detect_candidates(preprocess_frame(image))
                self.assertEqual(result.dice_values, expected)

    def test_masks_match_roi_and_exclude_pixels_outside_face(self):
        image = scene()
        draw_die(image, 350, 250, 4)
        image = cv2.warpAffine(image, cv2.getRotationMatrix2D((400, 300), 30, 1), (1000, 600), borderValue=(70,) * 3)
        candidate = detect_candidates(preprocess_frame(image)).dice[0]
        self.assertEqual(candidate.roi_bgr.shape, candidate.points_bgr.shape)
        self.assertEqual(candidate.roi_gray.shape, candidate.face_mask.shape)
        self.assertFalse(np.any(candidate.dark_mask[candidate.face_mask == 0]))
        self.assertTrue(set(np.unique(candidate.cleaned_mask)).issubset({0, 255}))
        self.assertEqual(sum(point.accepted for point in candidate.measurements), 4)

    def test_final_view_excludes_rejected_candidate_annotations(self):
        image = scene()
        draw_die(image, 50, 200, 0)
        draw_die(image, 250, 200, 4)
        prepared = preprocess_frame(image)
        detection = detect_candidates(prepared)
        self.assertIsNone(detection.candidates[0].die_id)
        self.assertEqual(detection.dice[0].die_id, 1)
        np.testing.assert_array_equal(detection.final_bgr[170:320, 20:175], prepared.resized_bgr[170:320, 20:175])
        self.assertFalse(np.array_equal(detection.annotated_bgr[170:320, 20:175], prepared.resized_bgr[170:320, 20:175]))

    def test_empty_scene_returns_zero_and_unchanged_final_image(self):
        prepared = preprocess_frame(scene())
        result = detect_candidates(prepared)
        self.assertEqual(result.dice_values, [])
        self.assertEqual(result.total_sum, 0)
        np.testing.assert_array_equal(result.final_bgr, prepared.resized_bgr)

    def test_invalid_filters_are_rejected(self):
        for values in [
            {"min_relative_area": 0.2, "max_relative_area": 0.1},
            {"min_aspect_ratio": 2, "max_aspect_ratio": 1},
            {"min_extent": 1.1}, {"min_extent": float("nan")},
            {"min_solidity": 1.1}, {"min_solidity": float("nan")},
            {"convex_face_mask": "yes"},
            {"min_rectangularity": 1.1}, {"min_rectangularity": float("nan")},
            {"separate_touching_dice": "yes"},
        ]:
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    CandidateFilters(**values)

    def test_color_annotations_do_not_change_preprocessing_images(self):
        image = scene()
        draw_die(image, 200, 200, 5)
        processed = preprocess_frame(image)
        original = processed.resized_bgr.copy()
        morphology = processed.morphology.copy()
        detection = detect_candidates(processed)
        np.testing.assert_array_equal(processed.resized_bgr, original)
        np.testing.assert_array_equal(processed.morphology, morphology)
        self.assertFalse(np.array_equal(detection.annotated_bgr, original))


if __name__ == "__main__":
    unittest.main()
