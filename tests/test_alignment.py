from gaze_query_rag.data.alignment import align_qa_with_gaze, build_alignment_report
from gaze_query_rag.data.chunking import chunk_paragraph
from gaze_query_rag.schemas import GazeRecord, QAExample


def _qa(example_id: str = "q1", paragraph_id: str = "p1") -> QAExample:
    return QAExample(
        example_id=example_id,
        paragraph_id=paragraph_id,
        paragraph_text="Alpha beta. Gamma delta.",
        question="What happened?",
        choices=["A", "B", "C", "D"],
        answer_index=0,
    )


def _gaze(reader_id: str, paragraph_id: str = "p1", word_index: int = 1) -> GazeRecord:
    return GazeRecord(
        reader_id=reader_id,
        paragraph_id=paragraph_id,
        word_index=word_index,
        word=f"w{word_index}",
        trt=10.0,
    )


def test_alignment_uses_title_level_question_metadata_fallback() -> None:
    qa = QAExample(
        example_id="hf1",
        paragraph_id="0",
        paragraph_text="HF paragraph text differs slightly.",
        question="Who found it?",
        choices=["A", "B", "C", "D"],
        answer_index=0,
        metadata={"raw": {"title": "Bottle Story", "level": 0}},
    )
    gaze = GazeRecord(
        reader_id="r1",
        paragraph_id="batch=1|article=2|paragraph=3|level=Adv|question=0",
        word_index=1,
        word="Bottle",
        trt=10.0,
        metadata={
            "article_title": "Bottle Story",
            "difficulty_level": "Adv",
            "question": "Who found it?",
            "paragraph": "OneStop paragraph text differs slightly.",
        },
    )

    aligned = align_qa_with_gaze([qa], [gaze])

    assert len(aligned) == 1
    assert list(aligned[0].reader_gaze) == ["r1"]


def test_alignment_report_is_generated() -> None:
    qa_examples = [_qa(), _qa("q2", "missing")]
    gaze_records = [_gaze("r1", word_index=2), _gaze("r1", word_index=1), _gaze("r2")]

    aligned = align_qa_with_gaze(qa_examples, gaze_records)
    report = build_alignment_report(aligned, qa_examples, gaze_records)

    assert len(aligned) == 1
    assert list(aligned[0].reader_gaze) == ["r1", "r2"]
    assert [record.word_index for record in aligned[0].reader_gaze["r1"]] == [1, 2]
    assert report["aligned_examples"] == 1
    assert report["unmatched_qa_examples"] == 1
    assert report["reader_coverage"]["matched_readers"] == 2


def test_chunk_paragraph_sentence_strategy() -> None:
    chunks = chunk_paragraph("p1", "One sentence. Second sentence!", "sentence", max_words=8)

    assert [chunk.text for chunk in chunks] == ["One sentence.", "Second sentence!"]
    assert chunks[0].char_start == 0
    assert chunks[1].paragraph_id == "p1"


def test_chunk_paragraph_fixed_words_with_stride() -> None:
    chunks = chunk_paragraph("p1", "one two three four five", "fixed_words", max_words=3, stride=1)

    assert [chunk.text for chunk in chunks] == ["one two three", "three four five"]
