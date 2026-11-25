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

## Установка

```bash
pip install -r requirements.txt
```

Готовые веса (`weights/detector_640.pt`) и тестовый набор (`testset/`) уже
в репозитории — для запуска системы датасет не нужен. Модель позы
`yolov8n-pose.pt` скачается автоматически при первом запуске.

## Запуск

```bash
# 1. Метрики уровня задачи на тестовом наборе (готовые веса из репозитория)
python src/evaluate_status.py --weights weights/detector_640.pt --kp-conf 0.3 --head-fallback

# 2. Сравнение с типовым решением (без оценки позы)
python src/evaluate_status.py --weights weights/detector_640.pt --baseline

# 3. Демо на тестовых изображениях (результат в runs/demo)
python src/run_inference.py --weights weights/detector_640.pt --source testset/images --out runs/demo

# 4. Демо на видео
python src/run_inference.py --weights weights/detector_640.pt --source video.mp4 --zone "0.5,0.6;1,0.6;1,1;0.5,1"
```

## Переобучение (опционально, нужен датасет)

```bash
# 1. Скачать датасет с HuggingFace (~1 ГБ)
python -c "from huggingface_hub import snapshot_download; snapshot_download('Voxel51/hard-hat-detection', repo_type='dataset', local_dir='data/hf_fallback/repo')"
cp data/hf_fallback/repo/samples.json data/hf_fallback/samples.json

# 2. Конвертация в формат YOLO (train/valid 75/25)
python src/convert_hf_dataset.py

# 3. Обучение детектора
python src/train_detector.py --data data/hard-hat-workers-hf/data.yaml --epochs 50 --name detect_n_640
```

## Статусы рабочего

`helmet_on` (норма), `helmet_in_hand`, `helmet_on_belt`, `no_helmet` (нарушения).
Уровни опасности: низкий / средний / критический — по положению ног
относительно опасной зоны камеры.
