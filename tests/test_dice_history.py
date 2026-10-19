"""Estatísticas por jogada, janela móvel e proteção contra frames repetidos."""

import unittest

from dice_history import DiceHistory, StableRollTracker


BOXES = ((100, 100, 40, 40), (200, 100, 40, 40))


class DiceHistoryTests(unittest.TestCase):
    def test_frequencies_count_each_roll_by_its_mean_and_include_tied_leaders(self):
        history = DiceHistory()
        self.assertIsNone(history.mean)
        self.assertEqual(history.most_frequent, ())
        history.record([1, 6])
        history.record([6, 3])
        self.assertEqual(history.frequencies, (0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0))
        self.assertEqual(history.mean, 4)
        self.assertEqual(history.most_frequent, (3.5, 4.5))
        history.record([3, 4])
        self.assertEqual(history.frequencies[5], 2)
        self.assertEqual(history.most_frequent, (3.5,))
        history.record([5, 4])
        self.assertEqual(history.most_frequent, (3.5, 4.5))

    def test_all_pairs_form_eleven_mean_groups_with_one_count_per_roll(self):
        history = DiceHistory()
        for first in range(1, 7):
            for second in range(1, 7):
                history.record([first, second])
        self.assertEqual(history.MEAN_VALUES, (1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6))
        self.assertEqual(history.frequencies, (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1))
        self.assertEqual(sum(history.frequencies), 36)
        self.assertEqual(history.mean, 3.5)
        self.assertEqual(history.most_frequent, (3.5,))

    def test_only_latest_fifty_rolls_contribute_to_statistics(self):
        history = DiceHistory()
        history.record([1, 1])
        for _ in range(50):
            history.record([6, 6])
        self.assertEqual(len(history.rolls), 50)
        self.assertEqual(history.rolls[0].number, 2)
        self.assertEqual(history.rolls[-1].number, 51)
        self.assertEqual(history.frequencies, (0,) * 10 + (50,))
        self.assertEqual(history.mean, 6)
        history.clear()
        self.assertEqual(history.frequencies, (0,) * 11)
        self.assertIsNone(history.mean)
        self.assertEqual(history.most_frequent, ())
        history.record([2, 2])
        self.assertEqual(history.rolls[0].number, 1)

    def test_incomplete_or_invalid_rolls_are_rejected(self):
        history = DiceHistory()
        for values in ([], [1], [1, 2, 3], [0, 2], [7, 2], [1.5, 2], [True, 2]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                history.record(values)
        self.assertEqual(len(history.rolls), 0)


class StableRollTrackerTests(unittest.TestCase):
    def test_stationary_dice_record_once_despite_repeated_frames_and_value_glitches(self):
        tracker = StableRollTracker()
        self.assertIsNone(tracker.observe([2, 6], BOXES, 0))
        self.assertIsNone(tracker.observe([2, 6], BOXES, .9))
        self.assertEqual(tracker.observe([2, 6], BOXES, 1), (2, 6))
        for now, values in [(2, [2, 6]), (3, [2]), (4, [3, 6]), (20, [2, 6])]:
            self.assertIsNone(tracker.observe(values, BOXES[:len(values)], now))
        tracker.cancel_pending()  # Pausar/retomar não deve duplicar a jogada.
        self.assertIsNone(tracker.observe([2, 6], BOXES, 30))

    def test_brief_absence_does_not_rearm_but_removal_allows_identical_next_roll(self):
        tracker = StableRollTracker()
        self.assertEqual(tracker.observe([3, 3], BOXES, 0, immediate=True), (3, 3))
        tracker.observe([], [], 1)
        tracker.observe([3, 3], BOXES, 1.2)
        self.assertFalse(tracker.armed)
        tracker.observe([], [], 2)
        tracker.observe([], [], 2.9)
        self.assertTrue(tracker.armed)
        self.assertIsNone(tracker.observe([3, 3], BOXES, 3))
        self.assertEqual(tracker.observe([3, 3], BOXES, 4), (3, 3))

    def test_moving_boxes_changing_values_and_partial_detection_restart_confirmation(self):
        tracker = StableRollTracker()
        tracker.observe([1, 6], BOXES, 0)
        moved = ((120, 100, 40, 40), BOXES[1])
        self.assertIsNone(tracker.observe([1, 6], moved, .9))
        self.assertIsNone(tracker.observe([4, 6], moved, 1.8))
        self.assertIsNone(tracker.observe([4], moved[:1], 2.5))
        self.assertIsNone(tracker.observe([4, 6], moved, 3))
        self.assertEqual(tracker.observe([4, 6], moved, 4), (4, 6))

    def test_manual_new_roll_and_photos_can_register_repeated_values(self):
        tracker = StableRollTracker()
        self.assertEqual(tracker.observe([5, 5], BOXES, 0, immediate=True), (5, 5))
        self.assertIsNone(tracker.observe([5, 5], BOXES, 1, immediate=True))
        tracker.new_roll()
        self.assertEqual(tracker.observe([5, 5], BOXES, 2, immediate=True), (5, 5))


if __name__ == "__main__":
    unittest.main()
