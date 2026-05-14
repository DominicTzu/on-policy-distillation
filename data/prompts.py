"""Prompt text for Stage A data construction."""


LONG_COT_SYSTEM_PROMPT = "You are a math reasoning teacher."

LONG_COT_USER_TEMPLATE = """Given a math word problem, solve it step by step and provide the final answer.

Requirements:
1. Show the complete reasoning process.
2. Include all necessary intermediate computations.
3. The final answer should be short and easy to parse.
4. Use exactly the following format:

### Long Rationale
...

### Answer
...

Question:
{question}"""


COMPRESSION_SYSTEM_PROMPT = (
    "You are a rationale compression module for math reasoning distillation."
)

COMPRESSION_USER_TEMPLATE = """Given a math problem, a teacher-generated long rationale, the gold final answer,
and a difficulty level, compress the rationale according to the difficulty level.

Compression policy:
- Easy: keep only one essential reasoning sentence.
- Medium: keep two concise reasoning sentences.
- Hard: keep all necessary intermediate computations, but remove repetition and self-reflection.

Requirements:
1. Preserve the reasoning needed to derive the final answer.
2. Remove redundant restatement, self-reflection, and unnecessary explanation.
3. Do not change the final answer.
4. The final answer must match the gold final answer.
5. Use exactly the following format:

### Rationale
...

### Answer
...

Question:
{question}

Teacher-generated long rationale:
{teacher_long_rationale}

Gold final answer:
{gold_answer}

Difficulty level:
{difficulty_level}"""


def build_long_cot_messages(question):
    return [
        {"role": "system", "content": LONG_COT_SYSTEM_PROMPT},
        {"role": "user", "content": LONG_COT_USER_TEMPLATE.format(question=question)},
    ]


def build_long_cot_prompt(question):
    return LONG_COT_SYSTEM_PROMPT + "\n\n" + LONG_COT_USER_TEMPLATE.format(
        question=question
    )


def build_compression_messages(record):
    user_prompt = COMPRESSION_USER_TEMPLATE.format(
        question=record["question"],
        teacher_long_rationale=record["teacher_long_rationale"],
        gold_answer=record["gold_answer"],
        difficulty_level=record["difficulty_level"],
    )
    return [
        {"role": "system", "content": COMPRESSION_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def build_compression_prompt(record):
    return COMPRESSION_SYSTEM_PROMPT + "\n\n" + COMPRESSION_USER_TEMPLATE.format(
        question=record["question"],
        teacher_long_rationale=record["teacher_long_rationale"],
        gold_answer=record["gold_answer"],
        difficulty_level=record["difficulty_level"],
    )
