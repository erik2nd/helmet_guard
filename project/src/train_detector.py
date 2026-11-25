# -*- coding: utf-8 -*-
"""Обучение детектора касок YOLOv8 на датасете Hard Hat Workers.

Пример:
    python src/train_detector.py --data data/hard-hat-workers/data.yaml \
        --model yolov8n.pt --epochs 50 --imgsz 640 --batch 8 --name detect_n_640
"""
import argparse
import json
from pathlib import Path

from ultralytics import YOLO


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="путь к data.yaml датасета")
    ap.add_argument("--model", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--name", default="detect_n_640")
    ap.add_argument("--project", default="runs")
    args = ap.parse_args()

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        patience=15,          # ранняя остановка
        workers=2,
        seed=42,
    )

    # Валидация лучших весов на val-выборке
    best = Path(model.trainer.save_dir) / "weights" / "best.pt"
    val = YOLO(str(best))
    m = val.val(data=args.data, imgsz=args.imgsz)

    metrics = {
        "model": args.model,
        "imgsz": args.imgsz,
        "epochs": args.epochs,
        "mAP50": float(m.box.map50),
        "mAP50_95": float(m.box.map),
        "precision": float(m.box.mp),
        "recall": float(m.box.mr),
        "per_class": {
            name: {
                "AP50": float(m.box.ap50[i]),
                "AP50_95": float(m.box.ap[i]),
                "precision": float(m.box.p[i]),
                "recall": float(m.box.r[i]),
            }
            for i, name in enumerate(m.names.values())
        },
    }
    out = Path(model.trainer.save_dir) / "val_metrics.json"
    out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("Лучшие веса:", best)


if __name__ == "__main__":
    main()
