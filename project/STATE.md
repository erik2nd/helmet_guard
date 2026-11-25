# Состояние проекта (для продолжения после паузы)

Проект: контроль ношения касок (YOLOv8n детекция + YOLOv8n-pose поза + собственный алгоритм).
Отчёты 1-2 готовы; делаются этапы 3 и 4. ВАЖНО: использовать Python 3.10
(`/c/Users/eriki/AppData/Local/Programs/Python/Python310/python.exe`), НЕ Python 3.12
(там другая версия ultralytics, resume ломается). Запуск с PYTHONIOENCODING=utf-8.

## Сделано

- Окружение: torch 2.11.0+cu128 (RTX 3050 Ti), ultralytics 8.4.216, cv2, pyarrow.
- Датасет: `project/data/hard-hat-workers-hf/` (HF-зеркало Hard Hat Workers,
  5000 изобр., helmet/head/person, split 3750/1250). Конвертер: convert_hf_dataset.py.
- Код: project/src/ (helmet_pipeline.py — алгоритм + TemporalFilter + head_fallback).
- Модель 640 ОБУЧЕНА (50 эпох, best=эпоха 40): `project/weights/detector_640.pt`.
  Val: helmet AP50 .958 / recall .916; head AP50 .920 / recall .885; person ~0 (как и планировалось).
  Метрики: `runs/detect/runs/detect_n_640/val_metrics.json`.
- Тестовый набор: `project/testset/` (47 изобр., 155 рабочих, gt.csv вручную).
- Оценка статусов (базовая cfg): acc 0.910, hidden_recall 1.0 (типовое решение: 0.0).
  Лучшая cfg (conf_det .25, kp_conf .3, head_fallback on): acc 0.916, hidden 1.0.
  JSON: runs/status_eval.json, status_eval_baseline.json, status_eval_best.json.
- Эксперименты E1-E5 (пороги): runs/experiments.csv. Выводы: conf_det>=0.4 убивает
  hidden recall; thr_head 0.15-0.25 плато; kp_conf 0.3 лучший; head_fallback +0.6%.
- Отчёт 3 этапа СГЕНЕРИРОВАН: `cv/Отчёт_по_3_этапу_контроль_СИЗ_на_стройке.docx`
  (генератор: src/make_report_stage3.py).

## В работе / следующие шаги

1. Идёт обучение YOLOv8n imgsz=960 (runs/detect/runs/detect_n_960), ~110 с/эпоха.
   По завершении train_detector.py сам запишет val_metrics.json в папку рана.
2. После обучения 960:
   - скопировать веса: cp runs/detect/runs/detect_n_960/weights/best.pt weights/detector_960.pt
   - оценка статусов: python src/evaluate_status.py --weights weights/detector_960.pt
     --kp-conf 0.3 --head-fallback --out runs/status_eval_960.json
   - сгенерировать отчёт 4: python src/make_report_stage4.py
     --m960 runs/detect/runs/detect_n_960/val_metrics.json
     --status960 runs/status_eval_960.json
   - ВНИМАНИЕ: в make_report_stage4.py вывод «960 не даёт выигрыша» — проверить по
     фактическим цифрам и переписать раздел 3, если 960 лучше!
3. Финальная уборка: удалить `cv/_preview`, обновить этот STATE.md.

## Заметки

- ultralytics складывает раны во вложенный `runs/detect/runs/<name>`.
- extra_inhand_01.png в тестсете — из train-пула (hard_hat_workers1604).
- baseline в evaluate_status.py = типовое решение (каски в руках/на поясе засчитываются надетыми).
