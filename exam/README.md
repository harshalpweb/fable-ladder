# Exam v1
The bar for the goal: 1,162 real questions (GPQA Diamond 198, MMLU-Pro 300, GSM8K 200, NQ-open 300, HumanEval 164),
half `dev` (open for development), half `hidden` (never read, only scored; a change stays only if the hidden half improves).

- `python3 exam_builder.py` rebuilds `exam_v1.jsonl` from public sources and checks it against the known hash.
- `python3 harness.py <system> [dev|hidden]` scores a system. A system is a file in `systems/` with `answer(item) -> str`.
- The question file is not stored in this repository (the GPQA authors ask that their questions not be posted openly).

Fable 5.2, closed book, first 30 dev questions of each: GPQA 30/30, MMLU-Pro 29/30, GSM8K 29/30, NQ-open 22/30
(strict exact match), HumanEval 30/30.
