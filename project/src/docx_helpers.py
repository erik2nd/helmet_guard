# -*- coding: utf-8 -*-
"""Помощник для генерации отчётов .docx в стиле отчётов 1-2 этапов."""
from docx import Document
from docx.shared import Cm, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH


def new_report():
    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Times New Roman"
    st.font.size = Pt(12)
    for section in doc.sections:
        section.left_margin = Cm(2.5)
        section.right_margin = Cm(1.5)
        section.top_margin = Cm(2)
        section.bottom_margin = Cm(2)
    return doc


def header_block(doc, stage, date_str):
    doc.add_heading(f"Отчёт по {stage} этапу проекта: Компьютерное зрение", level=1)
    doc.add_paragraph(date_str)
    doc.add_paragraph("Студенты: ____________________, ____________________")
    doc.add_paragraph(
        "Тема проекта: Контроль правильного ношения защитных касок на строительной "
        "площадке с оценкой опасности нарушения")
    doc.add_paragraph("2026 год, 1 семестр, дисциплина «Компьютерное зрение»")


def h2(doc, text):
    doc.add_heading(text, level=2)


def h3(doc, text):
    doc.add_heading(text, level=3)


def p(doc, text):
    doc.add_paragraph(text)


def bullets(doc, items):
    for it in items:
        doc.add_paragraph(it, style="List Bullet")


def table(doc, header, rows):
    t = doc.add_table(rows=1, cols=len(header))
    t.style = "Table Grid"
    for i, h in enumerate(header):
        t.rows[0].cells[i].text = str(h)
    for row in rows:
        cells = t.add_row().cells
        for i, v in enumerate(row):
            cells[i].text = str(v)
    return t


def picture(doc, path, width_cm=15, caption=None):
    doc.add_picture(str(path), width=Cm(width_cm))
    doc.paragraphs[-1].alignment = WD_ALIGN_PARAGRAPH.CENTER
    if caption:
        cap = doc.add_paragraph(caption)
        cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for run in cap.runs:
            run.font.size = Pt(10)
            run.italic = True
