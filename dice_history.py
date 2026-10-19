"""Histórico limitado de jogadas e confirmação de resultados estáveis."""

from collections import Counter, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class DiceRoll:
    number: int
    values: tuple[int, int]

    @property
    def mean(self):
        return sum(self.values) / 2


class DiceHistory:
    MEAN_VALUES = tuple(total / 2 for total in range(2, 13))

    def __init__(self, limit=50):
        self.rolls = deque(maxlen=limit)
        self.next_number = 1

    def record(self, values):
        values = tuple(values)
        if len(values) != 2 or any(type(value) is not int or not 1 <= value <= 6 for value in values):
            raise ValueError("Uma jogada precisa de dois valores inteiros entre 1 e 6.")
        self.rolls.append(DiceRoll(self.next_number, values))
        self.next_number += 1

    @property
    def frequencies(self):
        counts = Counter(roll.mean for roll in self.rolls)
        return tuple(counts[mean] for mean in self.MEAN_VALUES)

    @property
    def mean(self):
        count = len(self.rolls)
        return sum(roll.mean for roll in self.rolls) / count if count else None

    @property
    def most_frequent(self):
        counts = self.frequencies
        highest = max(counts)
        return tuple(mean for mean, count in zip(self.MEAN_VALUES, counts) if count == highest) if highest else ()

    def clear(self):
        self.rolls.clear()
        self.next_number = 1


class StableRollTracker:
    """Registra uma vez; rearma após ausência contínua dos dois dados.

    Uma perda parcial da detecção nunca rearma o registro. Valores e caixas
    precisam permanecer estáveis; pausas cancelam apenas a confirmação pendente.
    """

    def __init__(self, stable_seconds=1.0, empty_seconds=0.8, position_tolerance=6):
        self.stable_seconds = stable_seconds
        self.empty_seconds = empty_seconds
        self.position_tolerance = position_tolerance
        self.new_roll()

    def new_roll(self):
        self.armed = True
        self.cancel_pending()

    def cancel_pending(self):
        self.pending_values = None
        self.pending_boxes = None
        self.pending_since = None
        self.empty_since = None

    def observe(self, values, boxes, now, immediate=False):
        values, boxes = tuple(values), tuple(tuple(box) for box in boxes)
        if not values:
            self.pending_values = None
            self.pending_since = None
            if self.empty_since is None:
                self.empty_since = now
            if now - self.empty_since >= self.empty_seconds:
                self.new_roll()
            return None
        self.empty_since = None
        if not self.armed:
            return None
        if len(values) != 2 or len(boxes) != 2:
            self.cancel_pending()
            return None
        stable = (values == self.pending_values and self.pending_boxes is not None
                  and all(abs(a - b) <= self.position_tolerance
                          for box, anchor in zip(boxes, self.pending_boxes)
                          for a, b in zip(box, anchor)))
        if not stable:
            self.pending_values, self.pending_boxes, self.pending_since = values, boxes, now
        if immediate or now - self.pending_since >= self.stable_seconds:
            self.armed = False
            self.cancel_pending()
            return values
        return None
