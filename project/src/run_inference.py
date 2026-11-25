# -*- coding: utf-8 -*-
"""Запуск системы на изображениях или видео с визуализацией результата.

Примеры:
    python src/run_inference.py --weights runs/detect_n_640/weights/best.pt \
        --source testset/images --out runs/demo
    python src/run_inference.py --weights best.pt --source video.mp4 --zone "0.5,0.6;1,0.6;1,1;0.5,1"
"""
import argparse
from pathlib import Path

import cv2

from helmet_pipeline import HelmetMonitor, PipelineConfig, draw


def parse_zone(s: str):
    # "0.5,0.6;1,0.6;1,1;0.5,1" -> [(0.5, 0.6), ...]
    return [tuple(float(v) for v in p.split(",")) for p in s.split(";")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--pose", default="yolov8n-pose.pt")
    ap.add_argument("--source", required=True, help="папка с изображениями или видеофайл")
    ap.add_argument("--out", default="runs/demo")
    ap.add_argument("--zone", default="", help="опасная зона: 'x,y;x,y;...' в долях кадра")
    ap.add_argument("--conf-det", type=float, default=0.25)
    ap.add_argument("--max-frames", type=int, default=0, help="0 = без ограничений")
    args = ap.parse_args()

    cfg = PipelineConfig(conf_det=args.conf_det,
                         danger_zone=parse_zone(args.zone) if args.zone else [])
    monitor = HelmetMonitor(args.weights, args.pose, cfg)

    src = Path(args.source)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    if src.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        for p in sorted(src.iterdir()):
            if p.suffix.lower() not in exts:
                continue
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            workers = monitor.predict(frame)
            vis = draw(frame, workers, cfg)
            cv2.imwrite(str(out / p.name), vis)
            print(p.name, "->", [(w.status, w.danger) for w in workers])
    else:
        cap = cv2.VideoCapture(str(src))
        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vw = cv2.VideoWriter(str(out / (src.stem + "_out.mp4")),
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        n = 0
        while True:
            ok, frame = cap.read()
            if not ok or (args.max_frames and n >= args.max_frames):
                break
            workers = monitor.predict_video(frame)
            vw.write(draw(frame, workers, cfg))
            n += 1
            if n % 50 == 0:
                print(f"обработано кадров: {n}")
        cap.release()
        vw.release()
        print(f"готово, кадров: {n}, файл: {out / (src.stem + '_out.mp4')}")


if __name__ == "__main__":
    main()
