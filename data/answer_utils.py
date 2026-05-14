"""Answer parsing and normalization helpers."""

import re
from decimal import Decimal, InvalidOperation


ANSWER_MARKER = "### Answer"
RATIONALE_MARKER = "### Rationale"
LONG_RATIONALE_MARKER = "### Long Rationale"
NUMBER_RE = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?")


def split_rationale_and_answer(response):
    if not response:
        return None, None

    rationale_idx = response.find(RATIONALE_MARKER)
    answer_idx = response.find(ANSWER_MARKER)

    if rationale_idx < 0 or answer_idx < 0 or answer_idx < rationale_idx:
        return None, None

    rationale = response[rationale_idx + len(RATIONALE_MARKER) : answer_idx].strip()
    answer = response[answer_idx + len(ANSWER_MARKER) :].strip()
    return rationale, answer


def split_long_rationale_and_answer(response):
    if not response:
        return None, None

    rationale_idx = response.find(LONG_RATIONALE_MARKER)
    answer_idx = response.find(ANSWER_MARKER)

    if rationale_idx < 0 or answer_idx < 0 or answer_idx < rationale_idx:
        return None, None

    rationale = response[
        rationale_idx + len(LONG_RATIONALE_MARKER) : answer_idx
    ].strip()
    answer = response[answer_idx + len(ANSWER_MARKER) :].strip()
    return rationale, answer


def extract_answer(response):
    if not response:
        return None

    answer_idx = response.find(ANSWER_MARKER)
    if answer_idx >= 0:
        answer_text = response[answer_idx + len(ANSWER_MARKER) :].strip()
        for line in answer_text.splitlines():
            if line.strip():
                return line.strip()
        if answer_text:
            return answer_text

    lines = []
    for line in response.splitlines():
        if line.strip():
            lines.append(line.strip())
    if lines:
        return lines[-1]

    numbers = NUMBER_RE.findall(response)
    if numbers:
        return numbers[-1]
    return None


def extract_last_number(text):
    if not text:
        return None
    numbers = NUMBER_RE.findall(text)
    if not numbers:
        return None
    return numbers[-1]


def normalize_answer(answer):
    if answer is None:
        return None

    text = str(answer).strip()
    if not text:
        return None

    text = text.replace(",", "")
    text = text.replace("$", "")
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    text = text.rstrip(".")
    text = re.sub(r"^(answer\s*(is|:)?\s*)", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^=+\s*", "", text).strip()

    number = extract_last_number(text)
    if number:
        number = number.replace(",", "").replace("$", "")
        try:
            value = Decimal(number)
            if value == value.to_integral():
                return str(value.to_integral())
            return format(value.normalize(), "f")
        except InvalidOperation:
            pass

    return text.lower()


def answers_match(predicted, gold):
    normalized_predicted = normalize_answer(predicted)
    normalized_gold = normalize_answer(gold)
    if normalized_predicted is None or normalized_gold is None:
        return False
    return normalized_predicted == normalized_gold


def extract_gsm8k_gold_answer(solution):
    if not solution:
        return None
    if "####" in solution:
        return solution.split("####")[-1].strip()
    return extract_last_number(solution)


def extract_gsm8k_rationale(solution):
    if not solution:
        return ""
    if "####" in solution:
        return solution.split("####", 1)[0].strip()
    return solution.strip()
