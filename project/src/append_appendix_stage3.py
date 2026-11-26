# -*- coding: utf-8 -*-
"""Добавляет в отчёт 3 этапа приложение с исходным кодом и ссылками.

Приложение вставляется в конец существующего docx, содержимое отчёта
не изменяется. Запуск: python src/append_appendix_stage3.py
"""
from pathlib import Path

from docx import Document
from docx.shared import Pt

from docx_helpers import h3, p, table

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT.parent / "Отчёт_по_3_этапу_контроль_СИЗ_на_стройке.docx"
SRC = ROOT / "src"
REPO = "https://github.com/erik2nd/helmet_guard"

# (файл, назначение) — список со ссылками на GitHub
FILES = [
    ("helmet_pipeline.py", "ядро: привязка каски к частям тела, статусы, опасность, видеофильтр"),
    ("train_detector.py", "обучение детектора касок YOLOv8n"),
    ("evaluate_status.py", "оценка точности статусов на тестовом наборе, сравнение с типовым решением"),
    ("run_inference.py", "запуск системы на изображениях и видео"),
    ("run_experiments.py", "этап 4: перебор порогов, поиск лучшей конфигурации"),
    ("build_testset.py", "сборка тестового набора и шаблона разметки"),
    ("convert_hf_dataset.py", "конвертация датасета HuggingFace в формат YOLO"),
    ("test_pipeline_logic.py", "дымовой тест логики алгоритма"),
]

# Файлы, листинг которых включается в приложение
LISTINGS = ["helmet_pipeline.py", "train_detector.py", "evaluate_status.py"]


def add_code(doc, path):
    """Листинг файла: моноширинный шрифт 8 пт, без отступов."""
    for line in path.read_text(encoding="utf-8").splitlines():
        par = doc.add_paragraph()
        par.paragraph_format.space_after = Pt(0)
        par.paragraph_format.line_spacing = 1.0
        run = par.add_run(line if line else " ")
        run.font.name = "Courier New"
        run.font.size = Pt(8)


def main():
    doc = Document(str(REPORT))
    doc.add_page_break()

    doc.add_heading("Приложение А. Исходный код программной реализации", level=2)
    p(doc, "Полный исходный код проекта с историей изменений опубликован "
           f"в репозитории GitHub: {REPO}")
    p(doc, "Состав программной части (папка project/src):")
    table(doc,
          ["Файл", "Назначение", "Ссылка"],
          [(name, desc, f"{REPO}/blob/main/project/src/{name}")
           for name, desc in FILES])

    p(doc, "")
    p(doc, "Ниже приведены листинги ключевых модулей: ядра алгоритма, "
           "обучения детектора и оценки качества системы.")
    for name in LISTINGS:
        h3(doc, f"А.{LISTINGS.index(name) + 1}. Файл {name}")
        p(doc, f"{REPO}/blob/main/project/src/{name}")
        add_code(doc, SRC / name)

    doc.save(str(REPORT))
    print("Приложение добавлено:", REPORT)


if __name__ == "__main__":
    main()
