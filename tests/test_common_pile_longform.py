import json
from pathlib import Path

from prepare_common_pile_longform import grouped_documents, length_distribution


def record(doc_id, text, metadata=None, **extra):
    return {"id": doc_id, "text": text, "metadata": metadata or {}, **extra}


def test_doab_sections_are_joined_in_explicit_order():
    rows = [
        record("book.0", "zero", {"book_id": "book", "section_idx": 0}),
        record("book.2", "two", {"book_id": "book", "section_idx": 2}),
        record("other", "other", {"book_id": "other", "section_idx": 0}),
    ]
    docs = list(grouped_documents(rows, "doab_book_sections"))
    assert [(d.doc_id, d.text) for d in docs] == [
        ("book", "zero\n\ntwo"), ("other", "other")
    ]
    assert docs[0].metadata["component_count"] == 2


def test_pressbooks_and_usgpo_only_join_related_consecutive_parts():
    press = [
        record("u/1", "chapter 1", {"book_url": "book-a"}),
        record("u/2", "chapter 2", {"book_url": "book-a"}),
        record("v/1", "different", {"book_url": "book-b"}),
    ]
    assert [d.text for d in grouped_documents(press, "pressbooks_chapters")] == [
        "chapter 1\n\nchapter 2", "different"
    ]
    gpo = [
        record("CAL/CAL-pt1", "one"),
        record("CAL/CAL-pt2", "two"),
        record("BILL-1", "bill"),
    ]
    docs = list(grouped_documents(gpo, "usgpo_parts"))
    assert [(d.doc_id, d.text) for d in docs] == [
        ("CAL", "one\n\ntwo"), ("BILL-1", "bill")
    ]


def test_length_distribution_reports_long_context_thresholds():
    stats = length_distribution([10, 8193, 16385, 32769, 65537])
    assert stats["documents"] == 5
    assert stats["proportions_exceeding"]["8192"] == 0.8
    assert stats["proportions_exceeding"]["65536"] == 0.2
    assert stats["max"] == 65537
