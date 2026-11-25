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
    """Загружает ручную разметку тестового набора (testset/gt.csv).

    Возвращает список строк-словарей: {image, helmet_on, helmet_in_hand,
    helmet_on_belt, no_helmet}.
    """
    with open(testset / "gt.csv", encoding="utf-8") as f:
        return list(_csv.DictReader(f))


def main():
    """Перебор сеток параметров и поиск лучшей конфигурации (4 этап).

    Для каждой серии E1–E5 добавляются свои строки сетки: E1 — conf_det,
    E2 — thr_head, E3 — thr_hand=thr_belt, E4 — kp_conf, E5 —
    head_fallback. В каждой конфигурации меняется один параметр,
    остальные — из PipelineConfig по умолчанию; модели не переобучаются.
    Результаты пишутся в CSV (все столбцы параметров заполняются:
    неизменённые — значениями по умолчанию), в консоль печатается лучшая
    конфигурация по status_accuracy.
    """
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

    # Сетка экспериментов: каждая запись — одна конфигурация
    # (по одному изменённому параметру относительно базовой)
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
        # Одна конфигурация: PipelineConfig с переопределённым параметром
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
    # Недействующие в данной конфигурации параметры заполняются значениями
    # по умолчанию — в CSV всегда полный набор столбцов
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

    # Лучшая конфигурация по точности статуса (hidden_recall печатается рядом)
    best = max(rows, key=lambda r: r["status_accuracy"])
    print("Лучшая конфигурация:", best)


if __name__ == "__main__":
    main()
