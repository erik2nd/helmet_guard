# -*- coding: utf-8 -*-
"""Конвертация резервного датасета Voxel51/hard-hat-detection (HuggingFace)
в формат YOLO.

Источники: repo/data/*.png (изображения) + samples.json (разметка VOC).
Результат: data/hard-hat-workers-hf/{train,valid}/{images,labels} + data.yaml.
Разбиение 75/25, как в оригинальном датасете Hard Hat Workers.
"""
import json
import random
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "data"
SRC = ROOT / "hf_fallback"
DST = ROOT / "hard-hat-workers-hf"
CLASSES = ["helmet", "head", "person"]


def main():
    samples = json.load(open(SRC / "samples.json", encoding="utf-8"))["samples"]
    print("samples:", len(samples))

    for split in ("train", "valid"):
        (DST / split / "images").mkdir(parents=True, exist_ok=True)
        (DST / split / "labels").mkdir(parents=True, exist_ok=True)

    random.seed(42)
    random.shuffle(samples)
    n_train = int(len(samples) * 0.75)
    stats = {"train": 0, "valid": 0}
    cls_count = {c: 0 for c in CLASSES}
    skipped = 0

    for i, s in enumerate(samples):
        rel = s["filepath"]                  # data/hard_hat_workersN.png
        src_img = SRC / "repo" / rel
        if not src_img.exists():
            skipped += 1
            continue
        split = "train" if i < n_train else "valid"
        stem = Path(rel).stem
        shutil.copy(src_img, DST / split / "images" / f"{stem}.png")

        lines = []
        for det in s["ground_truth"]["detections"]:
            label = det["label"].lower()
            if label not in CLASSES:
                continue
            x, y, w, h = det["bounding_box"]  # VOC: левый верх + ширина/высота, норм.
            cx, cy = x + w / 2, y + h / 2
            lines.append(f"{CLASSES.index(label)} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            cls_count[label] += 1
        (DST / split / "labels" / f"{stem}.txt").write_text("\n".join(lines), encoding="utf-8")
        stats[split] += 1

    yaml = (f"train: {(DST / 'train' / 'images').as_posix()}\n"
            f"val: {(DST / 'valid' / 'images').as_posix()}\n"
            f"nc: {len(CLASSES)}\n"
            f"names: {CLASSES}\n")
    (DST / "data.yaml").write_text(yaml, encoding="utf-8")
    print("train:", stats["train"], "valid:", stats["valid"], "skipped:", skipped)
    print("boxes per class:", cls_count)
    print("data.yaml:", DST / "data.yaml")


if __name__ == "__main__":
    main()
