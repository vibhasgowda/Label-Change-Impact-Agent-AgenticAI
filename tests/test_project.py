import pytest
import fitz

from tools.label_reader import read_document
from tools.change_detector import (
    normalize_text,
    extract_sections,
    extract_changed_sections,
)
from tools.impact_assessment import assess_impact
from tools.task_generator import generate_task_list
from rag.inventory_loader import load_inventory


# --------------------------------------------------
# LABEL READER TESTS
# --------------------------------------------------

def test_read_txt_file(tmp_path):
    file = tmp_path / "sample.txt"
    file.write_text("Metformin label content", encoding="utf-8")

    result = read_document(str(file))

    assert result == "Metformin label content"


def test_read_pdf_file(tmp_path):
    file = tmp_path / "sample.pdf"

    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "Metformin pharmaceutical label")
    document.save(file)
    document.close()

    result = read_document(str(file))

    assert "Metformin pharmaceutical label" in result


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        read_document("nonexistent_label.pdf")


def test_unsupported_file_format(tmp_path):
    file = tmp_path / "sample.docx"
    file.write_text("Sample content", encoding="utf-8")

    with pytest.raises(ValueError):
        read_document(str(file))


# --------------------------------------------------
# CHANGE DETECTION TESTS
# --------------------------------------------------

def test_text_normalization():
    text = "WARNING   \n\n\n  Renal impairment"

    result = normalize_text(text)

    assert "\r" not in result
    assert "   " not in result
    assert result.strip() == result


def test_section_extraction():
    text = """
WARNINGS AND PRECAUTIONS
Monitor renal function regularly.

ADVERSE REACTIONS
Nausea and vomiting.
"""

    sections = extract_sections(text)

    assert "WARNINGS AND PRECAUTIONS" in sections
    assert "ADVERSE REACTIONS" in sections


def test_changed_section_detection():
    old_text = """
WARNINGS AND PRECAUTIONS
Monitor renal function annually.
"""

    new_text = """
WARNINGS AND PRECAUTIONS
Monitor renal function every three months.
"""

    changes = extract_changed_sections(old_text, new_text)

    assert len(changes) == 1
    assert changes[0]["section"] == "WARNINGS AND PRECAUTIONS"
    assert changes[0]["status"] == "Modified"


def test_no_changes_detected():
    text = """
WARNINGS AND PRECAUTIONS
Monitor renal function regularly.
"""

    changes = extract_changed_sections(text, text)

    assert changes == []


# --------------------------------------------------
# DOWNSTREAM INVENTORY TESTS
# --------------------------------------------------

def test_inventory_loading(tmp_path):
    folders = {
        "artwork": "Artwork",
        "website": "Website Content",
        "training": "Training Material",
        "country_submissions": "Country Submission",
    }

    for folder, category in folders.items():
        directory = tmp_path / folder
        directory.mkdir()

        (directory / "sample.txt").write_text(
            f"Sample {category} document",
            encoding="utf-8"
        )

    documents = load_inventory(str(tmp_path))

    assert len(documents) == 4

    categories = {document["category"] for document in documents}

    assert categories == set(folders.values())


def test_inventory_missing_directory():
    with pytest.raises(FileNotFoundError):
        load_inventory("nonexistent_inventory_directory")


def test_empty_inventory_files_are_ignored(tmp_path):
    directory = tmp_path / "training"
    directory.mkdir()

    (directory / "empty.txt").write_text("", encoding="utf-8")

    documents = load_inventory(str(tmp_path))

    assert documents == []


# --------------------------------------------------
# IMPACT ASSESSMENT VALIDATION TESTS
# --------------------------------------------------

def test_empty_detected_change():
    with pytest.raises(ValueError):
        assess_impact("", [])


def test_empty_retrieved_documents():
    result = assess_impact("A meaningful label change", [])

    assert result == "NO DOWNSTREAM IMPACTS IDENTIFIED"


# --------------------------------------------------
# TASK GENERATOR VALIDATION TESTS
# --------------------------------------------------

def test_empty_detected_changes():
    with pytest.raises(ValueError):
        generate_task_list("", "Some impact assessment")


def test_empty_impact_assessment():
    with pytest.raises(ValueError):
        generate_task_list("Some detected change", "")

