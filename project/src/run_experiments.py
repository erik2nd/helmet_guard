# -*- coding: utf-8 -*-
"""4 этап: исследование влияния параметров на качество системы.

Серии экспериментов на тестовом наборе (без переобучения моделей):
  E1. Порог уверенности детектора conf_det.
  E2. Порог привязки каски к голове thr_head.
  E3. Пороги привязки к рукам/поясу thr_hand, thr_belt.
  E4. Порог уверенности ключевых точек kp_conf.

Результат — CSV со строкой на эксперимент и лучшая конфигурация.
"""
import argparse
import csv
import itertools
from pathlib import Path

from evaluate_status import evaluate
from helmet_pipeline import HelmetMonitor, PipelineConfig

import csv as _csv


def load_gt(testset: Path):
    with open(testset / "gt.csv", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--testset", default="testset")
    ap.add_argument("--out", default="runs/experiments.csv")
    ap.add_argument("--series", default="E1,E2,E3,E4")
    args = ap.parse_args()

    testset = Path(args.testset)
    gt_rows = load_gt(testset)
    monitor = HelmetMonitor(args.weights, args.pose)

    grid = []
    series = set(args.series.split(","))
    if "E1" in series:
        grid += [{"conf_det": c} for c in (0.15, 0.25, 0.40, 0.60)]
    if "E2" in series:
        grid += [{"thr_head": t} for t in (0.10, 0.15, 0.20, 0.25)]
    if "E3" in series:
        grid += [{"thr_hand": t, "thr_belt": t} for t in (0.25, 0.35, 0.45)]
    if "E4" in series:
        grid += [{"kp_conf": k} for k in (0.3, 0.5, 0.7)]
    if "E5" in series:
        grid += [{"head_fallback": f} for f in (False, True)]

    rows = []
    for i, params in enumerate(grid, 1):
        cfg = PipelineConfig(**params)
        monitor.cfg = cfg
        m = evaluate(monitor, testset, gt_rows)
        row = dict(params)
        row.update(status_accuracy=round(m["status_accuracy"], 4),
                   hidden_recall=(round(m["hidden_violation_recall"], 4)
                                  if m["hidden_violation_recall"] is not None else ""),
                   workers=m["workers"])
        rows.append(row)
        print(f"[{i}/{len(grid)}] {params} -> acc={row['status_accuracy']}, "
              f"hidden_recall={row['hidden_recall']}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    keys = ["conf_det", "thr_head", "thr_hand", "thr_belt", "kp_conf",
            "head_fallback", "status_accuracy", "hidden_recall", "workers"]
    base = PipelineConfig()
    defaults = {"conf_det": base.conf_det, "thr_head": base.thr_head,
                "thr_hand": base.thr_hand, "thr_belt": base.thr_belt,
                "kp_conf": base.kp_conf, "head_fallback": base.head_fallback}
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=keys)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, defaults.get(k, "")) for k in keys})
    print("Сохранено:", out)

    best = max(rows, key=lambda r: r["status_accuracy"])
    print("Лучшая конфигурация:", best)


if __name__ == "__main__":
    main()
