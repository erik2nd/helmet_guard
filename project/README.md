# Контроль ношения защитных касок на стройке

Система: детектор касок/голов (YOLOv8n, дообучен на Hard Hat Workers)
+ оценка позы (YOLOv8n-pose, COCO) + собственный алгоритм привязки каски
к частям тела и оценки опасности нарушения.

## Структура

- `src/train_detector.py` — обучение и валидация детектора (mAP, Precision, Recall по классам)
- `src/helmet_pipeline.py` — собственный алгоритм: привязка каски к человеку
  и части тела, статус рабочего, оценка опасности по зоне, визуализация
- `src/run_inference.py` — запуск на папке изображений или видео
- `src/build_testset.py` — сборка тестового набора и контактных листов для разметки
- `src/evaluate_status.py` — метрики уровня задачи на тестовом наборе
  (точность статуса, полнота скрытых нарушений, сравнение с типовым решением)
- `src/run_experiments.py` — 4 этап: перебор порогов, поиск лучшей конфигурации
- `src/test_pipeline_logic.py` — дымовой тест логики алгоритма

## Запуск

```bash
# 1. Обучение детектора
python src/train_detector.py --data data/hard-hat-workers/data.yaml --epochs 50 --name detect_n_640

# 2. Метрики уровня задачи на тестовом наборе
python src/evaluate_status.py --weights runs/detect_n_640/weights/best.pt

# 3. Сравнение с типовым решением (без оценки позы)
python src/evaluate_status.py --weights runs/detect_n_640/weights/best.pt --baseline

# 4. Демо на изображениях/видео
python src/run_inference.py --weights runs/detect_n_640/weights/best.pt --source testset/images
```

## Статусы рабочего

`helmet_on` (норма), `helmet_in_hand`, `helmet_on_belt`, `no_helmet` (нарушения).
Уровни опасности: низкий / средний / критический — по положению ног
относительно опасной зоны камеры.
