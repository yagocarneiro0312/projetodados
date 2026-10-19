"""Testes dos pixels, proporções e conversões das etapas de pré-processamento."""

import unittest

import numpy as np
from PyQt6.QtGui import QColor, QImage

from qt_image_utils import gray_to_qimage, qimage_to_bgr
from vision_pipeline import preprocess_frame


class PreprocessingTests(unittest.TestCase):
    def test_standard_width_and_aspect_ratio_for_different_resolutions(self):
        for width, height in [(1920, 1080), (1280, 720), (640, 480), (320, 180), (1000, 750)]:
            with self.subTest(size=(width, height)):
                image = np.full((height, width, 3), 128, dtype=np.uint8)
                result = preprocess_frame(image)
                self.assertEqual(result.gray.shape[1], 1000)
                self.assertLessEqual(abs(result.gray.shape[0] - height * 1000 / width), 0.5)
                self.assertEqual(result.original_size, (width, height))
                self.assertEqual(result.resized_bgr.shape, (*result.gray.shape, 3))
                self.assertEqual(result.gray.dtype, np.uint8)
                np.testing.assert_array_equal(result.gray, 128)

    def test_known_bgr_colors_have_expected_grayscale_intensities(self):
        for bgr, expected in [((255, 0, 0), 29), ((0, 255, 0), 150), ((0, 0, 255), 76), ((0, 0, 0), 0), ((255, 255, 255), 255)]:
            with self.subTest(bgr=bgr):
                image = np.empty((12, 16, 3), dtype=np.uint8)
                image[:] = bgr
                np.testing.assert_array_equal(preprocess_frame(image).gray, expected)

    def test_processing_does_not_modify_source(self):
        image = np.arange(180, dtype=np.uint8).reshape(6, 10, 3)
        original = image.copy()
        preprocess_frame(image)
        np.testing.assert_array_equal(image, original)

    def test_threshold_boundary_is_strictly_greater_than_190(self):
        for intensity, expected in [(0, 0), (189, 0), (190, 0), (191, 255), (255, 255)]:
            with self.subTest(intensity=intensity):
                image = np.full((20, 1000, 3), intensity, dtype=np.uint8)
                result = preprocess_frame(image)
                np.testing.assert_array_equal(result.blur, intensity)
                np.testing.assert_array_equal(result.binary, expected)
                self.assertEqual(result.binary.shape, result.gray.shape)
                self.assertEqual(result.binary.dtype, np.uint8)

    def test_gaussian_blur_suppresses_isolated_bright_pixel_before_threshold(self):
        image = np.zeros((15, 1000, 3), dtype=np.uint8)
        image[7, 500] = 255
        result = preprocess_frame(image)
        self.assertEqual(result.gray[7, 500], 255)
        self.assertGreater(result.blur[7, 500], 0)
        self.assertLess(result.blur[7, 500], 190)
        self.assertEqual(np.count_nonzero(result.blur), 25)
        self.assertEqual(np.count_nonzero(result.binary), 0)

    def test_bright_face_is_white_and_dark_pip_and_table_are_black(self):
        image = np.full((400, 1000, 3), 120, dtype=np.uint8)
        image[100:300, 100:300] = 240
        y, x = np.ogrid[:400, :1000]
        image[(x - 200) ** 2 + (y - 200) ** 2 <= 16 ** 2] = 20
        result = preprocess_frame(image)
        self.assertEqual(result.binary[150, 150], 255)
        self.assertEqual(result.binary[200, 200], 0)
        self.assertEqual(result.binary[350, 350], 0)
        self.assertTrue(set(np.unique(result.binary)).issubset({0, 255}))

    def test_invalid_images_are_rejected(self):
        for image in [None, np.zeros((3, 3, 3), dtype=np.float32), np.zeros((3, 3), dtype=np.uint8), np.zeros((3, 3, 4), dtype=np.uint8), np.zeros((0, 3, 3), dtype=np.uint8)]:
            with self.subTest(image_type=type(image)):
                with self.assertRaises(ValueError):
                    preprocess_frame(image)

    def test_closing_fills_small_hole_without_overwriting_binary(self):
        image = np.zeros((400, 1000, 3), dtype=np.uint8)
        image[100:300, 100:300] = 255
        image[199:202, 199:202] = 0
        result = preprocess_frame(image)
        self.assertEqual(result.binary[200, 200], 0)
        self.assertEqual(result.closing[200, 200], 255)
        self.assertEqual(result.morphology[200, 200], 255)
        for mask in (result.closing, result.morphology):
            self.assertEqual(mask.shape, result.binary.shape)
            self.assertEqual(mask.dtype, np.uint8)
            self.assertTrue(set(np.unique(mask)).issubset({0, 255}))

    def test_opening_removes_small_isolated_object_and_preserves_face(self):
        image = np.zeros((400, 1000, 3), dtype=np.uint8)
        image[100:300, 100:300] = 255
        image[80:85, 700:705] = 255
        result = preprocess_frame(image)
        self.assertEqual(result.binary[82, 702], 255)
        self.assertEqual(result.closing[82, 702], 255)
        self.assertEqual(result.morphology[82, 702], 0)
        self.assertEqual(result.morphology[200, 200], 255)

    def test_two_closing_iterations_connect_nearby_regions(self):
        image = np.zeros((400, 1000, 3), dtype=np.uint8)
        image[100:300, 100:180] = 255
        image[100:300, 185:265] = 255
        result = preprocess_frame(image)
        self.assertEqual(result.binary[200, 181], 0)
        self.assertEqual(result.closing[200, 181], 255)
        self.assertEqual(result.morphology[200, 181], 255)

    def test_qimage_conversion_respects_bgr_order_and_padded_rows(self):
        image = QImage(5, 3, QImage.Format.Format_RGB888)
        image.fill(QColor("red"))
        image.setPixelColor(4, 2, QColor("blue"))
        bgr = qimage_to_bgr(image)
        self.assertEqual(bgr.shape, (3, 5, 3))
        np.testing.assert_array_equal(bgr[0, 0], [0, 0, 255])
        np.testing.assert_array_equal(bgr[2, 4], [255, 0, 0])
        image.fill(QColor("white"))
        np.testing.assert_array_equal(bgr[0, 0], [0, 0, 255])

    def test_gray_qimage_owns_pixels_and_accepts_non_contiguous_arrays(self):
        gray = np.array([[0, 10, 76, 20, 255, 30]], dtype=np.uint8)[:, ::2]
        image = gray_to_qimage(gray)
        gray[:] = 123
        self.assertEqual(image.format(), QImage.Format.Format_Grayscale8)
        self.assertEqual([image.pixelColor(x, 0).red() for x in range(3)], [0, 76, 255])


if __name__ == "__main__":
    unittest.main()
