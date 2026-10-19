"""Amostras aleatórias de dois dados, separadas das leituras da câmera."""

import csv
from dataclasses import dataclass
from pathlib import Path
from random import Random, SystemRandom

from dice_history import DiceHistory


THEORETICAL_PROBABILITIES = tuple(count / 36 for count in (1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1))


@dataclass(frozen=True)
class SimulationSample:
    seed: int
    history: DiceHistory

    def export_csv(self, path):
        with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.writer(stream, delimiter=";")
            writer.writerow(("origem", "semente", "jogada", "dado_1", "dado_2", "media"))
            for roll in self.history.rolls:
                writer.writerow(("simulacao", self.seed, roll.number, *roll.values,
                                 f"{roll.mean:g}".replace(".", ",")))


def generate_simulation(count=300, seed=None):
    """Dois sorteios uniformes independentes por jogada, sem ajustar a amostra."""
    if type(count) is not int or count < 1:
        raise ValueError("A quantidade de jogadas deve ser um inteiro positivo.")
    if seed is None:
        seed = SystemRandom().randrange(2**31)
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("A semente deve ser um inteiro entre 0 e 2147483647.")
    rng = Random(seed)
    history = DiceHistory(limit=None)
    for _ in range(count):
        history.record((rng.randint(1, 6), rng.randint(1, 6)))
    return SimulationSample(seed, history)


def create_simulation_figure(sample):
    """Figura exportável, com frequências observadas e probabilidades exatas."""
    from matplotlib.figure import Figure
    from matplotlib.ticker import MultipleLocator, PercentFormatter

    history = sample.history
    count = len(history.rolls)
    frequencies = [100 * frequency / count for frequency in history.frequencies]
    theoretical = [100 * probability for probability in THEORETICAL_PROBABILITIES]
    figure = Figure(figsize=(10, 5.8), dpi=110, layout="constrained", facecolor="white")
    axis = figure.add_subplot()
    bars = axis.bar(history.MEAN_VALUES, frequencies, width=.38, color="#2563eb",
                    label="Frequência simulada", zorder=3)
    axis.plot(history.MEAN_VALUES, theoretical, color="#d97706", linestyle="--",
              marker="o", markersize=4, linewidth=1.8, label="Distribuição teórica", zorder=4)
    axis.bar_label(bars, labels=[f"{value:.1f}%".replace(".", ",") for value in frequencies],
                   padding=5, fontsize=9, color="#1e293b", zorder=5,
                   bbox={"facecolor": "white", "edgecolor": "none", "pad": 1})
    axis.set_title(f"Média de dois dados — {count} jogadas simuladas", fontsize=15, pad=16)
    axis.set_xlabel(f"Média dos dois dados: (D1 + D2) ÷ 2\n"
                    f"Simulação computacional · Semente: {sample.seed}", fontsize=10, labelpad=10)
    axis.set_ylabel("Frequência relativa", fontsize=11)
    axis.set_xticks(history.MEAN_VALUES,
                      [f"{value:g}".replace(".", ",") for value in history.MEAN_VALUES])
    axis.set_ylim(0, max(25, max(frequencies) + 7))
    axis.yaxis.set_major_locator(MultipleLocator(5))
    axis.yaxis.set_major_formatter(PercentFormatter(100, decimals=0))
    axis.grid(axis="y", color="#e2e8f0", zorder=0)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper right", frameon=False, fontsize=9)
    return figure
