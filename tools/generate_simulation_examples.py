"""Exporta amostras de 200 e 300 jogadas e seus gráficos, sem selecionar resultados."""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dice_simulation import create_simulation_figure, generate_simulation


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, help="Semente opcional para reproduzir as duas amostras.")
    parser.add_argument("--output", type=Path, default=Path("docs/simulacao"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    sample = generate_simulation(300, args.seed)
    for count in (200, 300):
        current = generate_simulation(count, sample.seed) if count == 200 else sample
        stem = args.output / f"medias_{count}_jogadas"
        current.export_csv(stem.with_suffix(".csv"))
        create_simulation_figure(current).savefig(stem.with_suffix(".png"), dpi=300, facecolor="white")
        print(f"{count} jogadas simuladas; semente {current.seed}; frequencias {current.history.frequencies}")


if __name__ == "__main__":
    main()
