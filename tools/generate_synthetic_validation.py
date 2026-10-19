"""Gera PNGs e um relatório reproduzível do detector antigo e do robusto.

Executar na raiz: .venv/Scripts/python.exe tools/generate_synthetic_validation.py
"""

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from dice_detection import detect_candidates
from pipeline_parameters import emeet_calibration
from synthetic_dice import render_scene
from vision_pipeline import preprocess_frame


def run(output):
    output.mkdir(parents=True, exist_ok=True)
    six_output = output / "six"
    six_output.mkdir(exist_ok=True)
    parameters, faces, pips = emeet_calibration()
    old_parameters = replace(parameters, fill_face_holes=False)
    old_pips = replace(pips, threshold_mode="adaptive_gaussian", normalize_scale=False)
    results = []

    def evaluate(value, settings, group):
        image = render_scene(value=value, **settings)
        before = detect_candidates(preprocess_frame(image, old_parameters), faces, old_pips)
        after = detect_candidates(preprocess_frame(image, parameters), faces, pips)
        results.append(dict(group=group, expected=[value], settings=settings,
                            before=before.dice_values, after=after.dice_values))
        return image, before, after

    def cell(image, before, after, angle):
        canvas = np.full((188, 190, 3), (24, 20, 16), np.uint8)
        crop = image[238:322, 458:542].copy()
        for die in after.dice:
            cv2.drawContours(crop, [die.contour - np.array([[[458, 238]]])], -1, (80, 220, 80), 1)
            for pip in die.pips:
                center = (round(pip.center[0] - 458), round(pip.center[1] - 238))
                cv2.circle(crop, center, round(pip.radius), (255, 100, 30), 1)
        canvas[24:152, 31:159] = cv2.resize(crop, (128, 128), interpolation=cv2.INTER_NEAREST)
        cv2.putText(canvas, f"{angle} deg", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, .48, (230,) * 3, 1)
        old_text = ",".join(map(str, before.dice_values)) or "nenhum"
        new_text = ",".join(map(str, after.dice_values)) or "nenhum"
        cv2.putText(canvas, f"Antes: {old_text}", (8, 166), cv2.FONT_HERSHEY_SIMPLEX, .40,
                    (120, 160, 250) if before.dice_values != after.dice_values else (180,) * 3, 1)
        cv2.putText(canvas, f"Agora: {new_text}", (8, 182), cv2.FONT_HERSHEY_SIMPLEX, .40, (100, 230, 100), 1)
        return canvas

    six_cells = []
    for angle in range(0, 360, 15):
        settings = dict(side=36, pip_radius=.11, angle=angle)
        image, before, after = evaluate(6, settings, "six_full_turn")
        cv2.imwrite(str(six_output / f"six_{angle:03}.png"), image)
        six_cells.append(cell(image, before, after, angle))
    six_montage = np.vstack([np.hstack(six_cells[index:index + 6]) for index in range(0, 24, 6)])
    cv2.imwrite(str(output / "six_rotations.png"), six_montage)

    conditions = [dict(side=30, pip_radius=.10), dict(side=36, pip_radius=.11),
                  dict(side=36, pip_radius=.095, perspective=.6, blur=1),
                  dict(side=42, pip_radius=.082, perspective=1, blur=.8)]
    all_rows = []
    for value in range(1, 7):
        row = []
        for condition_index, settings in enumerate(conditions):
            for angle in range(0, 180, 15):
                image, before, after = evaluate(value, dict(settings, angle=angle), "all_faces")
                if condition_index == 1 and angle % 30 == 0:
                    row.append(cell(image, before, after, angle))
        all_rows.append(np.hstack(row))
    cv2.imwrite(str(output / "all_values.png"), np.vstack(all_rows))

    for value in range(1, 7):
        for center, angle in [((130, 110), 0), ((810, 120), 35), ((170, 430), 90), ((800, 420), 145)]:
            evaluate(value, dict(center=center, angle=angle, side=46, background=165, face=205,
                                 noise=1.2, perspective=.4), "positions_and_low_contrast")

    rng = np.random.default_rng(45619)
    for index in range(72):
        settings = dict(side=float(rng.uniform(34, 68)),
                        center=(float(rng.uniform(90, 910)), float(rng.uniform(90, 470))),
                        angle=float(rng.uniform(0, 360)), pip_radius=float(rng.uniform(.07, .10)),
                        perspective=float(rng.uniform(0, .5)), blur=float(rng.uniform(.3, .8)),
                        face=int(rng.integers(205, 246)), noise=.8)
        evaluate(index % 6 + 1, settings, "seeded_variations")

    summary = dict(total=len(results), before_correct=sum(r["before"] == r["expected"] for r in results),
                   after_correct=sum(r["after"] == r["expected"] for r in results))
    (output / "validation.json").write_text(json.dumps(dict(summary=summary, cases=results),
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary), flush=True)
    if summary["after_correct"] != summary["total"]:
        raise SystemExit("Há casos sintéticos com erro; consulte validation.json.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "docs" / "synthetic_dice")
    run(parser.parse_args().output)
