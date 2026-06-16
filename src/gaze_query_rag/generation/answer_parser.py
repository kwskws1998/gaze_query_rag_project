from __future__ import annotations

import re


def parse_choice(raw_output: str, num_choices: int) -> int | None:
    if num_choices <= 0 or num_choices > 26:
        raise ValueError("num_choices must satisfy 1 <= num_choices <= 26.")
    labels = [chr(ord("A") + index) for index in range(num_choices)]
    valid = set(labels)
    text = raw_output.strip()
    if not text:
        return None
    standalone = [match.group(1).upper() for match in re.finditer(r"\b([A-Za-z])\b", raw_output)]
    valid_standalone = [label for label in standalone if label in valid]
    if len(set(valid_standalone)) > 1:
        return None

    first_token = re.match(r"^\s*[\(\[]?([A-Za-z])[\)\].:\s,-]*", raw_output)
    if first_token:
        label = first_token.group(1).upper()
        if label in valid:
            return labels.index(label)

    patterns = [
        r"\banswer\s*(?:is|:)?\s*[\(\[]?([A-Za-z])[\)\]]?",
        r"\bchoice\s*[\(\[]?([A-Za-z])[\)\]]?",
        r"\boption\s*[\(\[]?([A-Za-z])[\)\]]?",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_output, flags=re.IGNORECASE)
        if match:
            label = match.group(1).upper()
            if label in valid:
                return labels.index(label)

    if len(valid_standalone) == 1:
        return labels.index(valid_standalone[0])
    return None
