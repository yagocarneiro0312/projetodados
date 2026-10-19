"""Amostras completas, reprodução e identificação da origem dos resultados."""

import csv
from pathlib import Path
import tempfile
import unittest

from dice_simulation import THEORETICAL_PROBABILITIES, generate_simulation


class DiceSimulationTests(unittest.TestCase):
    def test_samples_keep_every_roll_and_each_pair_contributes_once(self):
        for count in (200, 300):
            with self.subTest(count=count):
                sample = generate_simulation(count, seed=12345)
                self.assertEqual(len(sample.history.rolls), count)
                self.assertEqual(sum(sample.history.frequencies), count)
                self.assertEqual(sample.history.rolls[0].number, 1)
                self.assertEqual(sample.history.rolls[-1].number, count)
                for roll in sample.history.rolls:
                    self.assertTrue(all(1 <= value <= 6 for value in roll.values))

    def test_same_seed_reproduces_pairs_and_larger_sample_continues_the_sequence(self):
        first = generate_simulation(200, seed=12345)
        repeated = generate_simulation(200, seed=12345)
        larger = generate_simulation(300, seed=12345)
        different = generate_simulation(200, seed=54321)
        self.assertEqual(first.history.rolls, repeated.history.rolls)
        self.assertEqual(list(first.history.rolls), list(larger.history.rolls)[:200])
        self.assertNotEqual(first.history.rolls, different.history.rolls)

    def test_exact_probabilities_follow_the_discrete_triangular_distribution(self):
        self.assertAlmostEqual(sum(THEORETICAL_PROBABILITIES), 1)
        self.assertEqual(THEORETICAL_PROBABILITIES[5], 6 / 36)
        self.assertEqual(THEORETICAL_PROBABILITIES[0], 1 / 36)
        self.assertEqual(THEORETICAL_PROBABILITIES, THEORETICAL_PROBABILITIES[::-1])

    def test_export_preserves_all_pairs_seed_and_simulated_source(self):
        sample = generate_simulation(300, seed=12345)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "simulacao.csv"
            sample.export_csv(path)
            with path.open(encoding="utf-8-sig", newline="") as stream:
                rows = list(csv.DictReader(stream, delimiter=";"))
        self.assertEqual(len(rows), 300)
        for row, roll in zip(rows, sample.history.rolls):
            self.assertEqual(row["origem"], "simulacao")
            self.assertEqual(row["semente"], "12345")
            self.assertEqual((int(row["dado_1"]), int(row["dado_2"])), roll.values)
            self.assertEqual(float(row["media"].replace(",", ".")), roll.mean)

    def test_invalid_counts_and_seeds_are_rejected(self):
        for count in (0, -1, True, 2.5):
            with self.subTest(count=count), self.assertRaises(ValueError):
                generate_simulation(count)
        for seed in (-1, 2**31, True, "123"):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                generate_simulation(200, seed)


if __name__ == "__main__":
    unittest.main()
