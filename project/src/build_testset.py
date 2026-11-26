# -*- coding: utf-8 -*-
"""Сборка собственного тестового набора.

1) Копирует случайную выборку изображений из указанных источников
   (валидационная выборка датасета, другие папки) в testset/images.
2) Создаёт шаблон testset/gt.csv для ручной разметки статусов рабочих
   (без рамок — только число рабочих в каждом статусе).
3) Может построить контактные листы для удобной разметки.
"""
import argparse
import csv
import random
import shutil
from pathlib import Path

from PIL import Image, ImageDraw

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASSES = ["helmet_on", "helmet_in_hand", "helmet_on_belt", "no_helmet"]


def collect(sources):
    imgs = []
    for s in sources:
        p = Path(s)
        if p.is_dir():
            imgs += [f for f in p.rglob("*") if f.suffix.lower() in EXTS]
    return sorted(set(imgs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", required=True)
    ap.add_argument("--n", type=int, default=45)
    ap.add_argument("--out", default="testset")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--sheets", action="store_true", help="только построить контактные листы")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    if not args.sheets:
        imgs = collect(args.sources)
        random.seed(args.seed)
        sample = random.sample(imgs, min(args.n, len(imgs)))
        rows = []
        for i, p in enumerate(sample, 1):
            name = f"test_{i:03d}{p.suffix.lower()}"
            shutil.copy(p, out / "images" / name)
            rows.append({"image": name, **{c: 0 for c in CLASSES}})
        with open(out / "gt.csv", "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=["image"] + CLASSES)
            wr.writeheader()
            wr.writerows(rows)
        print(f"Скопировано {len(rows)} изображений в {out}/images, шаблон: {out}/gt.csv")

    # Контактные листы 3x3 с подписями имён для ручной разметки
    imgs = sorted((out / "images").iterdir())
    per, cell = 9, 320
    for si in range(0, len(imgs), per):
        batch = imgs[si:si + per]
        sheet = Image.new("RGB", (cell * 3, (cell + 20) * 3), "white")
        d = ImageDraw.Draw(sheet)
        for j, p in enumerate(batch):
            im = Image.open(p).convert("RGB")
            im.thumbnail((cell, cell))
            x, y = (j % 3) * cell, (j // 3) * (cell + 20)
            sheet.paste(im, (x, y + 20))
            d.text((x + 4, y + 3), p.name, fill="black")
        fn = out / f"sheet_{si // per + 1:02d}.jpg"
        sheet.save(fn, quality=85)
        print("лист:", fn)


if __name__ == "__main__":
    main()
