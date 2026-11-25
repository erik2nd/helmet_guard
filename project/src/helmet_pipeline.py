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
    """Один рабочий в кадре.

    Создаётся из результата модели оценки позы (рамка человека + 17
    ключевых точек), после чего алгоритм заполняет статус, уровень
    опасности и рамку привязанной каски.
    """
    bbox: tuple                       # рамка человека (x1, y1, x2, y2), пиксели
    kpts: np.ndarray                  # (17, 3): x, y, conf — ключевые точки COCO
    status: str = STATUS_UNKNOWN      # итоговый статус (заполняет compute_statuses)
    danger: int = DANGER_NONE         # уровень опасности (заполняет compute_danger)
    helmet_box: tuple | None = None   # рамка привязанной каски
    track_id: int = 0                 # id трека (для видео)

    @property
    def height(self) -> float:
        """Высота рамки человека в пикселях — «метр» для нормировки.

        У дальнего человека все расстояния в пикселях меньше, и деление
        на высоту делает пороги одинаковыми для ближних и дальних.
        max(..., 1.0) страхует от деления на ноль при вырожденной рамке.
        """
        return max(self.bbox[3] - self.bbox[1], 1.0)

    @property
    def feet(self) -> tuple:
        """Точка ног (x, y) — середина нижней грани рамки.

        Используется для проверки опасной зоны: важно, где стоит
        рабочий, а не где находится его туловище.
        """
        return ((self.bbox[0] + self.bbox[2]) / 2, self.bbox[3])

    def points(self, idxs, kp_conf):
        """Координаты видимых ключевых точек группы, массив (N, 2).

        idxs — список индексов точек COCO (например HAND_PTS), kp_conf —
        минимальная уверенность точки. Точки с низкой уверенностью
        отбрасываются, чтобы не притягивать каски к фантомным позициям.
        Если ни одна точка группы не видна, возвращается пустой массив
        (0, 2) — такая группа не участвует в привязке.
        """
        pts = [self.kpts[i][:2] for i in idxs if self.kpts[i][2] >= kp_conf]
        return np.array(pts) if pts else np.empty((0, 2))

    def head_points(self, kp_conf, fallback=False):
        """Точки головы; если лицо не видно (рабочий спиной) и включён
        fallback — оценка положения головы по плечам.

        Порядок работы:
        1. Берём видимые точки лица (нос, глаза, уши).
        2. Если они есть или fallback выключен — возвращаем результат.
        3. Иначе, если видны оба плеча, оцениваем голову как точку на
           0.18 высоты человека выше середины линии плеч (эмпирический
           коэффициент для типовой фигуры).
        Благодаря fallback рабочий спиной в каске не получает ложное
        «без каски».
        """
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
    """Все настраиваемые пороги конвейера в одном объекте.

    Значения по умолчанию — базовая конфигурация; оптимальные значения
    подобраны на 4 этапе (run_experiments.py): conf_det=0.25,
    thr_head=0.15, thr_hand=thr_belt=0.35, kp_conf=0.3,
    head_fallback=True.
    """
    conf_det: float = 0.25      # порог уверенности детектора
    conf_pose: float = 0.25     # порог уверенности детектора людей (pose)
    kp_conf: float = 0.5        # порог уверенности ключевой точки
    thr_head: float = THR_HEAD
    thr_hand: float = THR_HAND
    thr_belt: float = THR_BELT
    head_fallback: bool = False  # оценивать положение головы по плечам, если лицо не видно
    danger_zone: list = field(default_factory=list)  # многоугольник [(x, y), ...] в долях кадра


def _min_dist(point: np.ndarray, pts: np.ndarray) -> float:
    """Минимальное евклидово расстояние (пиксели) от точки до набора точек.

    Пустой набор (ни одна ключевая точка группы не видна) даёт inf —
    такая группа никогда не притянет каску.
    """
    if len(pts) == 0:
        return np.inf
    return float(np.min(np.linalg.norm(pts - point, axis=1)))


def assign_helmets(helmet_boxes, workers: list[Worker], cfg: PipelineConfig):
    """Привязывает каждую каску к человеку и части тела (жадное назначение).

    Возвращает список (helmet_box, worker_index, part), part ∈ {head, hand, belt}.

    Алгоритм:
    1. Для каждой каски берётся центр её рамки.
    2. Для каждого человека считаются нормированные расстояния от центра
       каски до трёх групп точек: голова, кисти, бёдра. Нормировка —
       деление на высоту человека, поэтому пороги одинаковы для ближних
       и дальних рабочих.
    3. Все пары «каска — (человек, часть тела)», прошедшие порог,
       сортируются по возрастанию расстояния.
    4. Пары назначаются жадно, начиная с самых близких: каждая каска
       попадает не более чем к одному человеку (used_h); у одного человека
       может быть только одна каска «на голове» (used_w), а касок «в руках»
       / «на поясе» может быть несколько (например, две каски в двух руках).
    """
    candidates = []
    for hb in helmet_boxes:
        cx, cy = (hb[0] + hb[2]) / 2, (hb[1] + hb[3]) / 2   # центр рамки каски
        c = np.array([cx, cy])
        for wi, w in enumerate(workers):
            h = w.height
            # Три группы точек с индивидуальными порогами (доли высоты)
            groups = (("head", w.head_points(cfg.kp_conf, cfg.head_fallback), cfg.thr_head),
                      ("hand", w.points(HAND_PTS, cfg.kp_conf), cfg.thr_hand),
                      ("belt", w.points(BELT_PTS, cfg.kp_conf), cfg.thr_belt))
            for part, pts, thr in groups:
                # Расстояние до ближайшей точки группы, нормированное на рост
                d = _min_dist(c, pts) / h
                if d < thr:
                    candidates.append((d, hb, wi, part))
    candidates.sort(key=lambda t: t[0])   # сначала самые «уверенные» привязки
    used_h, used_w, out = set(), set(), []
    for d, hb, wi, part in candidates:
        # Каска уже занята; «голова» у человека может быть занята другой каской
        if id(hb) in used_h or (wi, part) in used_w and part == "head":
            continue
        used_h.add(id(hb))
        used_w.add((wi, part))
        out.append((hb, wi, part))
    return out


def compute_statuses(workers: list[Worker], helmet_boxes, head_boxes, cfg: PipelineConfig):
    """Определяет статус каждого рабочего (по приоритету от лучшего к худшему).

    Для каждого рабочего:
    1. Каска привязана к голове  -> helmet_on (норма).
    2. Иначе привязана к кисти   -> helmet_in_hand (нарушение).
    3. Иначе привязана к бёдрам  -> helmet_on_belt (нарушение).
    4. Иначе ищем «голую голову»: рамка head детектора рядом с точками
       головы или в верхней трети рамки человека. Найдена -> no_helmet.
    5. Если головы не видно вовсе (рабочий вне кадра, спиной без
       fallback) -> unknown: система честно признаёт неуверенность.

    Изменяет объекты Worker на месте и возвращает тот же список.
    """
    assignments = assign_helmets(helmet_boxes, workers, cfg)
    by_worker = {}
    for hb, wi, part in assignments:
        by_worker.setdefault(wi, []).append((part, hb))

    for wi, w in enumerate(workers):
        parts = by_worker.get(wi, [])
        part_names = {p for p, _ in parts}   # куда привязались каски рабочего
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
                # Вариант 1: центр рамки head близко к точкам лица (порог
                # удвоен: точность положения головы по точкам ниже, чем
                # у каски, привязанной «сверху»)
                near_kpts = _min_dist(np.array([cx, cy]), head_pts) < cfg.thr_head * 2 * w.height
                # Вариант 2: центр рамки head в верхней трети рамки человека
                # (страховка на случай сбойной позы)
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
    """Уровень опасности нарушения по положению ног относительно опасной зоны.

    Зона — многоугольник cfg.danger_zone в долях кадра. Уровни:
    критический — ноги внутри зоны; средний — нарушение ближе к зоне,
    чем высота человека; низкий — дальше. Норма и unknown всегда
    DANGER_NONE. Если зона не задана, всем нарушителям ставится
    низкий уровень.
    """
    if not cfg.danger_zone:
        # Без зоны нет «географии»: любое нарушение — низкий уровень
        for w in workers:
            if w.status not in (STATUS_OK, STATUS_UNKNOWN):
                w.danger = DANGER_LOW
        return workers
    h_img, w_img = frame_shape[:2]
    # Вершины зоны из долей кадра переводятся в пиксели
    poly = np.array([(x * w_img, y * h_img) for x, y in cfg.danger_zone], dtype=np.float32)
    for w in workers:
        if w.status in (STATUS_OK, STATUS_UNKNOWN):
            w.danger = DANGER_NONE
            continue
        # measureDist=True: результат — расстояние до контура со знаком
        # (плюс внутри). d >= 0 значит ноги в зоне
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
        """n — длина скользящего окна в кадрах, k — сколько кадров с
        нарушением в окне нужно для подтверждения. По умолчанию «6 из 10»."""
        self.n, self.k = n, k
        self.history = {}  # track_id -> список последних статусов

    def update(self, track_id: int, status: str) -> str:
        """Обновляет историю статусов трека и возвращает статус для показа.

        Нарушение показывается, только если в последних n кадрах трека
        оно встречалось не менее k раз — одиночные ошибки детектора
        отсекаются. Возвращаемое значение: подтверждённый статус-нарушение,
        helmet_on для нормы или "pending" для ещё неподтверждённого
        нарушения (в подписи на кадре: «проверка...»).
        """
        h = self.history.setdefault(track_id, [])
        h.append(status)
        del h[:-self.n]   # оставляем только последние n статусов
        bad = sum(1 for s in h if s not in (STATUS_OK, STATUS_UNKNOWN))
        if bad >= self.k and status not in (STATUS_OK, STATUS_UNKNOWN):
            return status          # подтверждённое нарушение
        return STATUS_OK if status == STATUS_OK else "pending"


class HelmetMonitor:
    """Детектор касок + оценка позы + собственный алгоритм."""

    def __init__(self, det_weights, pose_weights="yolov8n-pose.pt", cfg: PipelineConfig | None = None):
        """Загружает обе модели: дообученный детектор касок/голов и
        предобученную pose-модель (веса скачиваются при первом запуске).
        cfg=None — параметры по умолчанию из PipelineConfig."""
        from ultralytics import YOLO
        self.det = YOLO(str(det_weights))
        self.pose = YOLO(str(pose_weights))
        self.cfg = cfg or PipelineConfig()
        self.class_names = self.det.names  # {0: 'head', 1: 'helmet', ...}
        self.temporal = TemporalFilter()

    def _det_boxes(self, frame):
        """Прогон детектора по кадру: разделение рамок на каски и головы.

        Возвращает (helmet_boxes, head_boxes) — списки (x1, y1, x2, y2).
        Рамки ниже conf_det отбрасывает сама модель; класс person
        игнорируется, людей находит pose-модель.
        """
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
        """Полный конвейер для одного кадра-изображения.

        Шаги: детекция касок/голов -> оценка позы -> сборка объектов
        Worker -> статусы -> опасность. Для видео используйте
        predict_video: там добавляются трекинг и временная фильтрация.
        """
        helmet_boxes, head_boxes = self._det_boxes(frame)
        pose_res = self.pose.predict(frame, conf=self.cfg.conf_pose, verbose=False)[0]

        # Каждый обнаруженный человек -> Worker с рамкой и 17 точками
        workers = []
        if pose_res.keypoints is not None:
            for box, kps in zip(pose_res.boxes, pose_res.keypoints.data):
                workers.append(Worker(bbox=tuple(box.xyxy[0].tolist()), kpts=kps.cpu().numpy()))

        compute_statuses(workers, helmet_boxes, head_boxes, self.cfg)
        compute_danger(workers, frame.shape, self.cfg)
        return workers

    def predict_video(self, frame: np.ndarray) -> list[Worker]:
        """Кадр видеопотока: трекинг людей + временная фильтрация нарушений.

        Отличия от predict: pose-модель вызывается в режиме трекинга
        ByteTrack (persist=True сохраняет id людей между кадрами), а
        каждый статус затем проходит через TemporalFilter — нарушение
        показывается, только если повторяется в 6 из 10 последних
        кадров трека.
        """
        helmet_boxes, head_boxes = self._det_boxes(frame)
        pose_res = self.pose.track(frame, conf=self.cfg.conf_pose, persist=True,
                                   tracker="bytetrack.yaml", verbose=False)[0]

        # id может отсутствовать, если трекер на этом кадре никого не закрепил
        workers = []
        if pose_res.keypoints is not None and pose_res.boxes.id is not None:
            for box, kps, tid in zip(pose_res.boxes, pose_res.keypoints.data,
                                     pose_res.boxes.id.tolist()):
                w = Worker(bbox=tuple(box.xyxy[0].tolist()), kpts=kps.cpu().numpy())
                w.track_id = int(tid)
                workers.append(w)

        compute_statuses(workers, helmet_boxes, head_boxes, self.cfg)
        for w in workers:
            # Фильтр по истории трека: нарушение могло ещё не подтвердиться
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
    """Визуализация: опасная зона, рамки рабочих с цветом по уровню опасности.

    На копию кадра рисуются: красный многоугольник опасной зоны; рамка
    каждого рабочего (цвет = уровень опасности: зелёный/жёлтый/оранжевый/
    красный) с подписью «статус | уровень»; жёлтые точки видимых ключевых
    точек; жёлтая рамка привязанной каски. Текст выводится через PIL,
    потому что cv2.putText не поддерживает кириллицу.
    """
    from PIL import Image, ImageDraw, ImageFont
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    d = ImageDraw.Draw(img)
    W, H = img.size
    # Системный шрифт с кириллицей (Windows-путь); если его нет — встроенный
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", max(14, H // 40))
    except OSError:
        font = ImageFont.load_default()

    if cfg.danger_zone:
        # Опасная зона: красный контур + точки в вершинах
        poly = [(x * W, y * H) for x, y in cfg.danger_zone]
        d.polygon(poly, outline=(255, 0, 0))
        for x, y in poly:
            d.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 0, 0))

    for w in workers:
        # Рамка и подпись в цвете уровня опасности (зелёный -> красный)
        color = DANGER_COLORS.get(w.danger, (200, 200, 200))
        x1, y1, x2, y2 = w.bbox
        d.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{STATUS_RU[w.status]} | {DANGER_RU.get(w.danger, '')}"
        d.text((x1 + 2, max(y1 - 20, 0)), label, fill=color, font=font)
        # Видимые ключевые точки (скелет) — жёлтым
        for kx, ky, kc in w.kpts:
            if kc >= cfg.kp_conf:
                d.ellipse([kx - 2, ky - 2, kx + 2, ky + 2], fill=(255, 255, 0))
        # Привязанная каска — жёлтой рамкой
        if w.helmet_box is not None:
            hx1, hy1, hx2, hy2 = w.helmet_box
            d.rectangle([hx1, hy1, hx2, hy2], outline=(255, 255, 0), width=2)
    # PIL рисует в RGB, cv2 ждёт BGR — конвертируем обратно
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
