# -*- coding: utf-8 -*-
"""Собственный алгоритм контроля ношения касок.

Объединяет детектор касок/голов (YOLOv8, дообучен на Hard Hat Workers)
и модель оценки позы (YOLOv8n-pose, предобучена на COCO):

1. Детекция: рамки классов helmet и head.
2. Оценка позы: рамки людей и 17 ключевых точек тела.
3. Привязка каски к человеку и части тела по нормированному расстоянию
   (расстояние до ближайшей ключевой точки группы, делённое на высоту
   человека в кадре).
4. Статус рабочего: «каска надета» / «в руках» / «на поясе» / «без каски».
5. Оценка опасности нарушения по положению ног относительно опасной зоны.
"""
from dataclasses import dataclass, field

import cv2
import numpy as np

# Индексы ключевых точек COCO
NOSE, L_EYE, R_EYE, L_EAR, R_EAR = 0, 1, 2, 3, 4
L_SHOULDER, R_SHOULDER = 5, 6
L_WRIST, R_WRIST = 9, 10
L_HIP, R_HIP = 11, 12

HEAD_PTS = [NOSE, L_EYE, R_EYE, L_EAR, R_EAR]
HAND_PTS = [L_WRIST, R_WRIST]
BELT_PTS = [L_HIP, R_HIP]

# Пороги нормированного расстояния (доля высоты человека)
THR_HEAD = 0.15   # каска у головы
THR_HAND = 0.35   # каска у кисти
THR_BELT = 0.35   # каска у бедра

STATUS_OK = "helmet_on"
STATUS_HAND = "helmet_in_hand"
STATUS_BELT = "helmet_on_belt"
STATUS_NONE = "no_helmet"
STATUS_UNKNOWN = "unknown"

STATUS_RU = {
    STATUS_OK: "каска надета",
    STATUS_HAND: "каска в руках",
    STATUS_BELT: "каска на поясе",
    STATUS_NONE: "без каски",
    STATUS_UNKNOWN: "не определён",
    "pending": "проверка...",
}

DANGER_NONE, DANGER_LOW, DANGER_MID, DANGER_CRIT = -1, 0, 1, 2
DANGER_RU = {DANGER_NONE: "норма", DANGER_LOW: "низкий", DANGER_MID: "средний", DANGER_CRIT: "критический"}


@dataclass
class Worker:
    bbox: tuple                       # рамка человека (x1, y1, x2, y2)
    kpts: np.ndarray                  # (17, 3): x, y, conf
    status: str = STATUS_UNKNOWN
    danger: int = DANGER_NONE
    helmet_box: tuple | None = None   # рамка привязанной каски
    track_id: int = 0                 # id трека (для видео)

    @property
    def height(self) -> float:
        return max(self.bbox[3] - self.bbox[1], 1.0)

    @property
    def feet(self) -> tuple:
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    def points(self, idxs, kp_conf):
        """Видимые ключевые точки группы как массив (N, 2)."""
        pts = [self.kpts[i][:2] for i in idxs if self.kpts[i][2] >= kp_conf]
        return np.array(pts) if pts else np.empty((0, 2))

    def head_points(self, kp_conf, fallback=False):
        """Точки головы; если лицо не видно (рабочий спиной) и включён
        fallback — оценка положения головы по плечам."""
        pts = self.points(HEAD_PTS, kp_conf)
        if len(pts) or not fallback:
            return pts
        sh = self.points([L_SHOULDER, R_SHOULDER], kp_conf)
        if len(sh) == 0:
            return pts
        center = sh.mean(axis=0)
        return np.array([[center[0], center[1] - 0.18 * self.height]])


@dataclass
class PipelineConfig:
    conf_det: float = 0.25      # порог уверенности детектора
    conf_pose: float = 0.25     # порог уверенности детектора людей (pose)
    kp_conf: float = 0.5        # порог уверенности ключевой точки
    thr_head: float = THR_HEAD
    thr_hand: float = THR_HAND
    thr_belt: float = THR_BELT
    head_fallback: bool = False  # оценивать положение головы по плечам, если лицо не видно
    danger_zone: list = field(default_factory=list)  # многоугольник [(x, y), ...] в долях кадра


def _min_dist(point: np.ndarray, pts: np.ndarray) -> float:
    if len(pts) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(pts - point, axis=1)))


def assign_helmets(helmet_boxes, workers: list[Worker], cfg: PipelineConfig):
    """Привязывает каждую каску к человеку и части тела.

    Возвращает список (helmet_box, worker_index, part), part ∈ {head, hand, belt}.
    Каска привязывается не более чем к одному человеку (жадно, по возрастанию
    нормированного расстояния).
    """
    candidates = []
    for hb in helmet_boxes:
        cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
        c = np.array([cx, cy])
        for wi, w in enumerate(workers):
            h = w.height
            groups = (("head", w.head_points(cfg.kp_conf, cfg.head_fallback), cfg.thr_head),
                      ("hand", w.points(HAND_PTS, cfg.kp_conf), cfg.thr_hand),
                      ("belt", w.points(BELT_PTS, cfg.kp_conf), cfg.thr_belt))
            for part, pts, thr in groups:
                d = _min_dist(c, pts) / h
                if d < thr:
                    candidates.append((d, hb, wi, part))
    candidates.sort(key=lambda t: t[0])
    used_h, used_w, out = set(), set(), []
    for d, hb, wi, part in candidates:
        if id(hb) in used_h or (wi, part) in used_w and part == "head":
            continue
        used_h.add(id(hb))
        used_w.add((wi, part))
        out.append((hb, wi, part))
    return out


def compute_statuses(workers: list[Worker], helmet_boxes, head_boxes, cfg: PipelineConfig):
    """Определяет статус каждого рабочего."""
    assignments = assign_helmets(helmet_boxes, workers, cfg)
    by_worker = {}
    for hb, wi, part in assignments:
        by_worker.setdefault(wi, []).append((part, hb))

    for wi, w in enumerate(workers):
        parts = by_worker.get(wi, [])
        part_names = {p for p, _ in parts}
        head_pts = w.head_points(cfg.kp_conf, cfg.head_fallback)
        head_visible = len(head_pts) > 0

        if "head" in part_names:
            w.status = STATUS_OK
            w.helmet_box = next(hb for p, hb in parts if p == "head")
        elif "hand" in part_names:
            w.status = STATUS_HAND
            w.helmet_box = next(hb for p, hb in parts if p == "hand")
        elif "belt" in part_names:
            w.status = STATUS_BELT
            w.helmet_box = next(hb for p, hb in parts if p == "belt")
        else:
            # Голова без каски: рамка head детектора рядом с точками головы
            # или в верхней трети рамки человека.
            found_head = False
            for hb in head_boxes:
                cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2
                near_kpts = _min_dist(np.array([cx, cy]), head_pts) < cfg.thr_head * 2 * w.height
                in_top = (w.bbox[0] <= cx <= w.bbox[2]) and (w.bbox[1] <= cy <= w.bbox[1] + 0.35 * w.height)
                if near_kpts or in_top:
                    found_head = True
                    break
            if found_head or head_visible:
                w.status = STATUS_NONE
            else:
                w.status = STATUS_UNKNOWN
    return workers


def compute_danger(workers: list[Worker], frame_shape, cfg: PipelineConfig):
    """Уровень опасности нарушения по положению ног относительно опасной зоны."""
    if not cfg.danger_zone:
        for w in workers:
            if w.status not in (STATUS_OK, STATUS_UNKNOWN):
                w.danger = DANGER_LOW
        return workers
    h_img, w_img = frame_shape[:2]
    poly = np.array([(x * w_img, y * h_img) for x, y in cfg.danger_zone], dtype=np.float32)
    for w in workers:
        if w.status in (STATUS_OK, STATUS_UNKNOWN):
            w.danger = DANGER_NONE
            continue
        d = cv2.pointPolygonTest(poly, w.feet, True)
        if d >= 0:
            w.danger = DANGER_CRIT
        elif abs(d) < w.height:
            w.danger = DANGER_MID
        else:
            w.danger = DANGER_LOW
    return workers


class TemporalFilter:
    """Фильтрация нарушений по времени для видео: нарушение фиксируется,
    если статус-нарушение повторяется в k из последних n кадров трека."""

    def __init__(self, n=10, k=6):
        self.n, self.k = n, k
        self.history = {}  # track_id -> список последних статусов

    def update(self, track_id: int, status: str) -> str:
        h = self.history.setdefault(track_id, [])
        h.append(status)
        del h[:-self.n]
        bad = sum(1 for s in h if s not in (STATUS_OK, STATUS_UNKNOWN))
        if bad >= self.k and status not in (STATUS_OK, STATUS_UNKNOWN):
            return status          # подтверждённое нарушение
        return STATUS_OK if status == STATUS_OK else "pending"


class HelmetMonitor:
    """Детектор касок + оценка позы + собственный алгоритм."""

    def __init__(self, det_weights, pose_weights="yolov8n-pose.pt", cfg: PipelineConfig | None = None):
        from ultralytics import YOLO
        self.det = YOLO(str(det_weights))
        self.pose = YOLO(str(pose_weights))
        self.cfg = cfg or PipelineConfig()
        self.class_names = self.det.names  # {0: 'head', 1: 'helmet', ...}
        self.temporal = TemporalFilter()

    def _det_boxes(self, frame):
        det_res = self.det.predict(frame, conf=self.cfg.conf_det, verbose=False)[0]
        helmet_boxes, head_boxes = [], []
        for box in det_res.boxes:
            cls = self.class_names[int(box.cls)]
            xyxy = tuple(box.xyxy[0].tolist())
            if cls == "helmet":
                helmet_boxes.append(xyxy)
            elif cls == "head":
                head_boxes.append(xyxy)
        return helmet_boxes, head_boxes

    def predict(self, frame: np.ndarray) -> list[Worker]:
        helmet_boxes, head_boxes = self._det_boxes(frame)
        pose_res = self.pose.predict(frame, conf=self.cfg.conf_pose, verbose=False)[0]

        workers = []
        if pose_res.keypoints is not None:
            for box, kps in zip(pose_res.boxes, pose_res.keypoints.data):
                workers.append(Worker(bbox=tuple(box.xyxy[0].tolist()), kpts=kps.cpu().numpy()))

        compute_statuses(workers, helmet_boxes, head_boxes, self.cfg)
        compute_danger(workers, frame.shape, self.cfg)
        return workers

    def predict_video(self, frame: np.ndarray) -> list[Worker]:
        """Кадр видеопотока: трекинг людей + временная фильтрация нарушений."""
        helmet_boxes, head_boxes = self._det_boxes(frame)
        pose_res = self.pose.track(frame, conf=self.cfg.conf_pose, persist=True,
                                   tracker="bytetrack.yaml", verbose=False)[0]

        workers = []
        if pose_res.keypoints is not None and pose_res.boxes.id is not None:
            for box, kps, tid in zip(pose_res.boxes, pose_res.keypoints.data,
                                     pose_res.boxes.id.tolist()):
                w = Worker(bbox=tuple(box.xyxy[0].tolist()), kpts=kps.cpu().numpy())
                w.track_id = int(tid)
                workers.append(w)

        compute_statuses(workers, helmet_boxes, head_boxes, self.cfg)
        for w in workers:
            w.status = self.temporal.update(w.track_id, w.status)
        compute_danger(workers, frame.shape, self.cfg)
        return workers


DANGER_COLORS = {
    DANGER_NONE: (0, 200, 0),
    DANGER_LOW: (0, 220, 220),
    DANGER_MID: (0, 140, 255),
    DANGER_CRIT: (0, 0, 255),
}


def draw(frame: np.ndarray, workers: list[Worker], cfg: PipelineConfig) -> np.ndarray:
    """Визуализация: опасная зона, рамки рабочих с цветом по уровню опасности."""
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    W, H = img.size
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(14, H // 40))
    except OSError:
        font = ImageFont.load_default()

    if cfg.danger_zone:
        poly = [(x * W, y * H) for x, y in cfg.danger_zone]
        d.polygon(poly, outline=(255, 0, 0))
        for x, y in poly:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))

    for w in workers:
        color = DANGER_COLORS.get(w.danger, (200, 200, 200))
        x1, y1, x2, y2 = w.bbox
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{STATUS_RU[w.status]} | {DANGER_RU.get(w.danger, '')}"
        d.text((x1 + 2, max(y1 - 20, 0)), label, fill=color, font=font)
        for kx, ky, kc in w.kpts:
            if kc >= cfg.kp_conf:
                d.ellipse([kx - 2, ky - 2, kx + 2, ky + 2], fill=(255, 255, 0))
        if w.helmet_box is not None:
            hx1, hy1, hx2, hy2 = w.helmet_box
            d.rectangle([hx1, hy1, hx2, hy2], outline=(255, 255, 0), width=2)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
