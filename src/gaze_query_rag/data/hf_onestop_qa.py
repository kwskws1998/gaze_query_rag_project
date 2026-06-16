from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

from gaze_query_rag.schemas import QAExample, SchemaInferenceError


LOCAL_FALLBACK = Path("OneStop-Eye-Movements/data_preprocessing/onestop_qa.json")


def _single_match(columns: Iterable[str], candidates: list[str], field_name: str) -> str:
    columns_list = list(columns)
    lower_to_original = {column.lower(): column for column in columns_list}
    exact = [lower_to_original[name] for name in candidates if name in lower_to_original]
    if len(exact) == 1:
        return exact[0]
    if len(exact) > 1:
        raise SchemaInferenceError(f"Ambiguous {field_name} columns: {exact}")
    fuzzy = [
        column
        for column in columns_list
        if any(candidate in column.lower() for candidate in candidates)
    ]
    if len(fuzzy) == 1:
        return fuzzy[0]
    raise SchemaInferenceError(f"Could not infer {field_name} column from {columns_list}")


def infer_qa_schema(columns: list[str]) -> dict[str, str]:
    schema = {
        "paragraph": _single_match(
            columns, ["paragraph", "context", "passage", "paragraph_text"], "paragraph"
        ),
        "question": _single_match(columns, ["question"], "question"),
    }
    if "answers" in columns:
        schema["choices"] = "answers"
    elif "choices" in columns:
        schema["choices"] = "choices"
    elif "options" in columns:
        schema["choices"] = "options"
    else:
        answer_cols = [f"answer_{idx}" for idx in range(1, 5) if f"answer_{idx}" in columns]
        if len(answer_cols) >= 2:
            schema["choices"] = "|".join(answer_cols)
        else:
            raise SchemaInferenceError(f"Could not infer choices columns from {columns}")
    for name in ["answer_index", "label", "correct_answer", "answer"]:
        if name in columns:
            schema["answer"] = name
            break
    if "answer" not in schema:
        schema["answer"] = ""
    for name in ["example_id", "id", "question_id", "q_ind"]:
        if name in columns:
            schema["example_id"] = name
            break
    for name in ["paragraph_id", "para_id"]:
        if name in columns:
            schema["paragraph_id"] = name
            break
    return schema


def _canonical_local_paragraph_id(
    article_id: str, paragraph_id: Any, level: str, question_id: Any
) -> str:
    if "_" in str(article_id):
        article_batch, article_in_batch = str(article_id).split("_", 1)
    else:
        article_batch, article_in_batch = "", str(article_id)
    return (
        f"batch={article_batch}|article={article_in_batch}|paragraph={paragraph_id}"
        f"|level={level}|question={question_id}"
    )


def load_local_onestop_qa_json(path: str | Path) -> list[QAExample]:
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8") as fh:
        payload = json.load(fh)
    articles = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
    if not isinstance(articles, list):
        raise SchemaInferenceError("Local OneStop QA JSON must contain a list or a top-level data list.")

    examples: list[QAExample] = []
    for article in articles:
        article_id = str(article["article_id"])
        for paragraph in article.get("paragraphs", []):
            paragraph_number = paragraph.get("paragraph_id")
            qas = paragraph.get("qas", [])
            for level, level_payload in paragraph.items():
                if level in {"paragraph_id", "qas"} or not isinstance(level_payload, dict):
                    continue
                context = level_payload.get("context")
                if not context:
                    continue
                for qa in qas:
                    q_ind = qa.get("q_ind")
                    choices = [str(choice) for choice in qa.get("answers", [])]
                    if not choices:
                        raise SchemaInferenceError(f"QA item has no answers: {qa}")
                    paragraph_key = _canonical_local_paragraph_id(
                        article_id, paragraph_number, level, q_ind
                    )
                    example_id = f"local:{article_id}:{paragraph_number}:{level}:{q_ind}"
                    metadata = {
                        "source": str(source_path),
                        "article_id": article_id,
                        "article_title": article.get("title"),
                        "paragraph_number": paragraph_number,
                        "difficulty_level": level,
                        "raw_qa": qa,
                    }
                    examples.append(
                        QAExample(
                            example_id=example_id,
                            paragraph_id=paragraph_key,
                            paragraph_text=str(context),
                            question=str(qa["question"]),
                            choices=choices,
                            answer_index=0,
                            metadata=metadata,
                        )
                    )
    return examples


def _coerce_choices(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if isinstance(value, tuple):
        return [str(item) for item in value]
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("["):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    return [str(item) for item in parsed]
            except json.JSONDecodeError:
                pass
        return [part.strip() for part in stripped.split("|") if part.strip()]
    raise SchemaInferenceError(f"Unsupported choices value: {value!r}")


def _coerce_answer_index(value: Any) -> int | None:
    if value in (None, ""):
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    text = str(value).strip()
    if not text:
        return None
    if text.isdigit():
        return int(text)
    if len(text) == 1 and text.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        return ord(text.upper()) - ord("A")
    return None


def _fallback_path() -> Path | None:
    env_path = os.environ.get("ONESTOP_QA_JSON")
    if env_path:
        return Path(env_path)
    path = Path.cwd() / LOCAL_FALLBACK
    return path if path.exists() else None


def load_onestop_qa(
    dataset_name: str = "malmaud/onestop_qa",
    split: str | None = None,
    cache_dir: str | Path | None = None,
) -> list[QAExample]:
    candidate_path = Path(dataset_name)
    if candidate_path.exists():
        return load_local_onestop_qa_json(candidate_path)
    try:
        from datasets import load_dataset

        dataset = load_dataset(dataset_name, cache_dir=str(cache_dir) if cache_dir else None)
        selected_split = split or next(iter(dataset.keys()))
        rows = dataset[selected_split]
        columns = list(rows.column_names)
        schema = infer_qa_schema(columns)
        examples: list[QAExample] = []
        for idx, row in enumerate(rows):
            choices_source = schema["choices"]
            if "|" in choices_source:
                choices = [str(row[col]) for col in choices_source.split("|")]
            else:
                choices = _coerce_choices(row[choices_source])
            answer_index = (
                _coerce_answer_index(row[schema["answer"]]) if schema.get("answer") else None
            )
            if answer_index is None and str(dataset_name) == "malmaud/onestop_qa":
                answer_index = 0
            paragraph_id = str(row.get(schema.get("paragraph_id", ""), idx))
            example_id = str(row.get(schema.get("example_id", ""), idx))
            examples.append(
                QAExample(
                    example_id=example_id,
                    paragraph_id=paragraph_id,
                    paragraph_text=str(row[schema["paragraph"]]),
                    question=str(row[schema["question"]]),
                    choices=choices,
                    answer_index=answer_index,
                    metadata={"source": dataset_name, "split": selected_split, "raw": dict(row)},
                )
            )
        return examples
    except Exception as exc:
        fallback = _fallback_path()
        if fallback is None:
            raise RuntimeError(
                f"Could not load Hugging Face dataset {dataset_name!r}, and no local fallback exists."
            ) from exc
        return load_local_onestop_qa_json(fallback)
