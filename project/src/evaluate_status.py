# -*- coding: utf-8 -*-
"""Оценка итогового статуса рабочего на собственном тестовом наборе.

Разметка тестового набора (testset/gt.csv), без рамок — только статусы:
    image,helmet_on,helmet_in_hand,helmet_on_belt,no_helmet
    img01.jpg,2,0,0,1        # 2 рабочих в касках, 1 без каски

Сравнение мультимножеств предсказанных и истинных статусов по изображению:
верно распознанными считаются min(pred[c], gt[c]) рабочих каждого класса.

Метрики: точность статуса, матрица ошибок (по избыткам/недостаткам),
полнота выявления скрытых нарушений (каска в руках / на поясе).
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from helmet_pipeline import (HelmetMonitor, PipelineConfig,
                             STATUS_BELT, STATUS_HAND, STATUS_NONE, STATUS_OK)

CLASSES = [STATUS_OK, STATUS_HAND, STATUS_BELT, STATUS_NONE]
HIDDEN = {STATUS_HAND, STATUS_BELT}


def count_statuses(workers):
    c = Counter(w.status for w in workers)
    return {k: c.get(k, 0) for k in CLASSES}


def evaluate(monitor, testset: Path, gt_rows, baseline=False):
    """baseline=True — типовое решение без оценки позы: каска на кадре у
    человека = «надета» (каски в руках/на поясе не различаются)."""
    total_gt, correct = 0, 0
    confusion = Counter()       # (истинный, предсказанный) по несовпавшим
    hidden_gt, hidden_found = 0, 0
    per_image = []

    for row in gt_rows:
        img_path = testset / "images" / row["image"]
        frame = cv2.imread(str(img_path))
        if frame is None:
            print("не найдено:", img_path)
            continue
        workers = monitor.predict(frame)
        pred = count_statuses(workers)
        if baseline:
            # Типовой детектор: любая каска, привязанная к человеку, — «надета»
            moved = pred[STATUS_HAND] + pred[STATUS_BELT]
            pred[STATUS_OK] += moved
            pred[STATUS_HAND] = pred[STATUS_BELT] = 0
        gt = {k: int(row[k]) for k in CLASSES}

        n_gt = sum(gt.values())
        n_ok = sum(min(pred[k], gt[k]) for k in CLASSES)
        total_gt += n_gt
        correct += n_ok

        for k in CLASSES:  # избытки и недостатки как пары путаницы
            miss = gt[k] - min(pred[k], gt[k])
            if miss > 0:
                extra = max(sum(pred.values()) - n_ok, 0)
                pred_major = max((p for p in CLASSES if p != k),
                                 key=lambda p: pred[p] - gt[p], default=None)
                for _ in range(miss):
                    confusion[(k, pred_major if extra > 0 else "пропуск")] += 1

        h_gt = gt[STATUS_HAND] + gt[STATUS_BELT]
        h_pred = min(pred[STATUS_HAND], gt[STATUS_HAND]) + min(pred[STATUS_BELT], gt[STATUS_BELT])
        # нарушение признано нарушением, даже если «в руках» перепутано с «на поясе»
        h_pred += min(max(pred[STATUS_HAND] - gt[STATUS_HAND], 0), gt[STATUS_BELT] - min(pred[STATUS_BELT], gt[STATUS_BELT]))
        h_pred += min(max(pred[STATUS_BELT] - gt[STATUS_BELT], 0), gt[STATUS_HAND] - min(pred[STATUS_HAND], gt[STATUS_HAND]))
        hidden_gt += h_gt
        hidden_found += min(h_pred, h_gt)

        per_image.append({"image": row["image"], "gt": gt, "pred": pred})

    metrics = {
        "images": len(per_image),
        "workers": total_gt,
        "status_accuracy": correct / total_gt if total_gt else 0.0,
        "hidden_violation_recall": hidden_found / hidden_gt if hidden_gt else None,
        "hidden_violations_gt": hidden_gt,
        "confusion": {f"{a}->{b}": n for (a, b), n in confusion.most_common()},
        "per_image": per_image,
    }
    return metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True, help="веса детектора (best.pt)")
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--testset", default="testset")
    ap.add_argument("--out", default="runs/status_eval.json")
    ap.add_argument("--baseline", action="store_true", help="типовое решение без позы")
    ap.add_argument("--thr-head", type=float, default=0.15)
    ap.add_argument("--thr-hand", type=float, default=0.35)
    ap.add_argument("--thr-belt", type=float, default=0.35)
    ap.add_argument("--conf-det", type=float, default=0.25)
    ap.add_argument("--kp-conf", type=float, default=0.5)
    ap.add_argument("--head-fallback", action="store_true",
                    help="оценивать положение головы по плечам, если лицо не видно")
    args = ap.parse_args()

    testset = Path(args.testset)
    with open(testset / "gt.csv", encoding="utf-8") as f:
        gt_rows = list(csv.DictReader(f))

    cfg = PipelineConfig(conf_det=args.conf_det, kp_conf=args.kp_conf,
                         thr_head=args.thr_head, thr_hand=args.thr_hand,
                         thr_belt=args.thr_belt, head_fallback=args.head_fallback)
    monitor = HelmetMonitor(args.weights, args.pose, cfg)
    metrics = evaluate(monitor, testset, gt_rows, baseline=args.baseline)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Изображений: {metrics['images']}, рабочих: {metrics['workers']}")
    print(f"Точность статуса: {metrics['status_accuracy']:.3f}")
    if metrics["hidden_violation_recall"] is not None:
        print(f"Полнота скрытых нарушений: {metrics['hidden_violation_recall']:.3f}")
    print("Путаница:", metrics["confusion"])


if __name__ == "__main__":
    main()
