# -*- coding: utf-8 -*-
"""Дымовой тест логики алгоритма на синтетических данных (без моделей)."""
import numpy as np

from helmet_pipeline import (PipelineConfig, Worker, assign_helmets,
                             compute_danger, compute_statuses,
                             STATUS_OK, STATUS_HAND, STATUS_BELT, STATUS_NONE,
                             DANGER_CRIT, DANGER_LOW, DANGER_MID, DANGER_NONE)


def make_worker(bbox, kpts_conf=0.9):
    """bbox=(x1,y1,x2,y2); ключевые точки расставляются схематично по рамке."""
    x1, y1, x2, y2 = bbox
    h = y2 - y1
    cx = (x1 + x2) / 2
    k = np.zeros((17, 3))
    # нос, глаза, уши — верх рамки
    for i, dy in ((0, .05), (1, .03), (2, .03), (3, .04), (4, .04)):
        k[i] = [cx + (i - 2) * 4, y1 + dy * h, kpts_conf]
    # кисти — середина по бокам
    k[9] = [x1 + .1 * (x2 - x1), y1 + .55 * h, kpts_conf]
    k[10] = [x2 - .1 * (x2 - x1), y1 + .55 * h, kpts_conf]
    # бёдра
    k[11] = [cx - .08 * (x2 - x1), y1 + .6 * h, kpts_conf]
    k[12] = [cx + .08 * (x2 - x1), y1 + .6 * h, kpts_conf]
    return Worker(bbox=bbox, kpts=k)


def box_around(x, y, s=20):
    return (x - s, y - s, x + s, y + s)


def main():
    cfg = PipelineConfig()
    # Рабочий 1: каска на голове. Рабочий 2: каска в руке. Рабочий 3: без каски.
    w1 = make_worker((100, 50, 220, 400))
    w2 = make_worker((400, 50, 520, 400))
    w3 = make_worker((700, 50, 820, 400))
    workers = [w1, w2, w3]

    nose1 = w1.kpts[0][:2]
    wrist2 = w2.kpts[9][:2]
    helmets = [box_around(*nose1), box_around(*wrist2)]
    heads = [box_around(*w3.kpts[0][:2])]  # рамка head у третьего

    compute_statuses(workers, helmets, heads, cfg)
    st = [w.status for w in workers]
    print("статусы:", st)
    assert st == [STATUS_OK, STATUS_HAND, STATUS_NONE], st

    # Каска на поясе
    w4 = make_worker((1000, 50, 1120, 400))
    hip4 = w4.kpts[11][:2]
    compute_statuses([w4], [box_around(*hip4)], [], cfg)
    print("пояс:", w4.status)
    assert w4.status == STATUS_BELT

    # Опасная зона: низ кадра справа
    cfg2 = PipelineConfig(danger_zone=[(0.8, 0.8), (1, 0.8), (1, 1), (0.8, 1)])
    w3.status = STATUS_NONE
    w6 = make_worker((100, 50, 220, 400))   # далеко от зоны
    w6.status = STATUS_NONE
    # ноги w3: x=760, y=400 при кадре 1200x500 -> (0.633, 0.8) — рядом с зоной
    compute_danger([w3, w1, w6], (500, 1200, 3), cfg2)
    print("опасность w3 (рядом с зоной):", w3.danger,
          "| w1 (норма):", w1.danger, "| w6 (далеко):", w6.danger)
    assert w1.danger == DANGER_NONE       # нарушения нет — опасности нет
    assert w6.danger == DANGER_LOW        # нарушение вдали от зоны
    assert w3.danger in (DANGER_MID, DANGER_CRIT)

    # Рабочий спиной: точки головы не видны, каска на голове -> неопределён/без каски
    w5 = make_worker((100, 50, 220, 400), kpts_conf=0.9)
    w5.kpts[:5, 2] = 0.1  # точки лица не видны
    compute_statuses([w5], [box_around(*w5.kpts[0][:2])], [], cfg)
    print("спиной, каска на голове, точки лица не видны:", w5.status)

    # То же с fallback по плечам -> каска должна распознаться как надетая
    w5.kpts[5, :] = [130, 110, 0.9]   # левое плечо
    w5.kpts[6, :] = [190, 110, 0.9]   # правое плечо
    cfg_fb = PipelineConfig(head_fallback=True)
    compute_statuses([w5], [box_around(160, 75)], [], cfg_fb)
    print("спиной + fallback по плечам:", w5.status)
    assert w5.status == STATUS_OK

    print("ВСЕ ПРОВЕРКИ ПРОШЛИ")


if __name__ == "__main__":
    main()
