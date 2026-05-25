from __future__ import annotations

import argparse
import json
import logging
import re
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

from datasets import Dataset, concatenate_datasets

from data.mmlupro import MMLUPro
from utils import load_vllm_llm, prompt_vllm


LOGGER = logging.getLogger(__name__)
LANGUAGES = ["english", "hindi", "bengali", "kannada", "tamil"]
ANSWER_RE = re.compile(r"####\s*ANSWER\s*:\s*\(?([A-J])\)?", re.IGNORECASE)
REASONING_BLOCK_RE = re.compile(
    r"<reasoning>(.*?)</reasoning>",
    re.IGNORECASE | re.DOTALL,
)


def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )


def _options_to_text(options: list[str]) -> str:
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return "\n".join(
        f"({letters[idx]}) {choice}" for idx, choice in enumerate(options)
    )


def sample_datasets(
        samples_per_language: list[int], seed: int = 42
) -> Dataset:
    language_codes = ["en", "hi", "bn", "kn", "ta"]
    subsets = []

    for idx, language in enumerate(LANGUAGES):
        n = samples_per_language[idx]
        if n == 0:
            continue
        dataset = MMLUPro(language=language_codes[idx]).get_unified_dataset()
        n = min(n, len(dataset))
        dataset = dataset.shuffle(seed=seed)
        dataset = dataset.select(range(n))
        LOGGER.info("Sampled %d rows for %s", n, language)
        subsets.append(dataset)

    combined = concatenate_datasets(subsets)
    return combined.shuffle(seed=seed)


def format_teacher_prompt(instruction: str, language: str, gold_answer: str) -> str:

    lang_names = {
        "en": "English",
        "hi": "Hindi",
        "bn": "Bengali",
        "kn": "Kannada",
        "ta": "Tamil",
    }
    lang_name = lang_names.get(language, "English")
    return (
        f"You are an expert. Answer the following question in {lang_name}.\n"
        f"Think step by step before answering.\n\n"
        f"Write your reasoning inside <reasoning>...</reasoning> tags.\n"
        f"The correct answer is ({gold_answer}). "
        f"Explain step by step why ({gold_answer}) is correct.\n"
        f"At the end, write exactly: #### ANSWER: ({gold_answer})\n\n"
        f"{instruction}"
    )


def detect_language(text: str) -> str:
    counts = {
        "hi": 0,  
        "bn": 0,  
        "kn": 0,  
        "ta": 0,  
        "en": 0,  
    }
    for char in text:
        cp = ord(char)
        if 0x0900 <= cp <= 0x097F:
            counts["hi"] += 1
        elif 0x0980 <= cp <= 0x09FF:
            counts["bn"] += 1
        elif 0x0C80 <= cp <= 0x0CFF:
            counts["kn"] += 1
        elif 0x0B80 <= cp <= 0x0BFF:
            counts["ta"] += 1
        elif char.isascii() and char.isalpha():
            counts["en"] += 1
    return max(counts, key=lambda k: counts[k])


def is_valid_generation(parsed: dict[str, str]) -> tuple[bool, str]:

    if parsed["final_answer"] == "":
        return False, "missing_answer"

    if parsed["reasoning"] == "":
        return False, "missing_reasoning"

    return True, "ok"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Query teacher and build train/val/test corpus JSONL"
    )
    parser.add_argument(
        "--teacher_model",
        required=True,
        help="Hugging Face path to the teacher model",
    )
    parser.add_argument(
        "--num_samples",
        type=str,
        required=True,
        help=(
            "Comma-separated sample counts for english,hindi,bengali,"
            "kannada,tamil"
        ),
    )
    parser.add_argument(
        "--output_file",
        required=True,
        help="Output JSONL path for train corpus",
    )
    parser.add_argument(
        "--val_output_file",
        default="data/val.jsonl",
        help="Output JSONL path for validation corpus",
    )
    parser.add_argument(
        "--test_output_file",
        default="data/test.jsonl",
        help="Output JSONL path for test corpus (split equally from val)",
    )
    parser.add_argument(
        "--val_per_language",
        type=int,
        default=200,
        help=(
            "Total held-out samples per language before val/test split. "
            "Half go to val, half go to test (default 200 → 100 val + 100 test each)"
        ),
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--gpu_memory_utilization",
        type=float,
        default=0.6,
        help="Target fraction of GPU memory for vLLM; lower if startup fails",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="vLLM tensor parallel size",
    )
    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def _parse_num_samples(raw_value: str) -> list[int]:
    parts = [part.strip() for part in raw_value.split(",") if part.strip()]
    if len(parts) != len(LANGUAGES):
        raise ValueError(
            "--num_samples must contain exactly 5 comma-separated integers "
            "for english,hindi,bengali,kannada,tamil"
        )
    try:
        counts = [int(part) for part in parts]
    except ValueError as exc:
        raise ValueError("--num_samples must contain only integers") from exc
    if any(count < 0 for count in counts):
        raise ValueError("--num_samples values must be >= 0")
    return counts


def _build_instruction(row: dict[str, Any]) -> str:
    options = row["options"]
    if not isinstance(options, list):
        options = list(options)
    return f"{row['question']}\n\n{_options_to_text(options)}"


def _make_eval_record(row: dict[str, Any]) -> dict:
    return {
        "question": _build_instruction(row),
        "gold_answer": str(row.get("answer", "")).upper()[:1],
        "language": row["language"],
        "subject": row.get("subject"),
    }


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    samples_per_language = _parse_num_samples(args.num_samples)

    sampled = sample_datasets(
        samples_per_language=samples_per_language,
        seed=args.seed,
    )
    LOGGER.info("Collected %d samples total", len(sampled))

    teacher, tokenizer = load_vllm_llm(
        model_id=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
    )

    rows = list(sampled)
    random.seed(args.seed)
    random.shuffle(rows)

    language_buckets: dict[str, list] = defaultdict(list)
    for row in rows:
        language_buckets[row["language"]].append(row)

    train_rows: list = []
    val_rows:   list = []
    test_rows:  list = []

    half = args.val_per_language // 2  

    for lang, lang_rows in language_buckets.items():
        held_out  = lang_rows[:args.val_per_language]
        remaining = lang_rows[args.val_per_language:]

        val_rows.extend(held_out[:half])   
        test_rows.extend(held_out[half:])  
        train_rows.extend(remaining)

    random.shuffle(val_rows)
    random.shuffle(test_rows)

    LOGGER.info(
        "Split: %d train | %d val | %d test",
        len(train_rows), len(val_rows), len(test_rows),
    )
    for lang in sorted(language_buckets):
        LOGGER.info(
            "  %-10s train=%d  val=%d  test=%d",
            lang,
            sum(1 for r in train_rows if r["language"] == lang),
            sum(1 for r in val_rows   if r["language"] == lang),
            sum(1 for r in test_rows  if r["language"] == lang),
        )

    val_path  = Path(args.val_output_file)
    test_path = Path(args.test_output_file)

    _write_jsonl(val_path,  [_make_eval_record(r) for r in val_rows])
    _write_jsonl(test_path, [_make_eval_record(r) for r in test_rows])
    LOGGER.info("Saved %d val  rows to %s", len(val_rows),  val_path)
    LOGGER.info("Saved %d test rows to %s", len(test_rows), test_path)

    LOGGER.info("Building gold-conditioned prompts for %d train rows...", len(train_rows))
    prompts = []
    for row in train_rows:
        gold = str(row.get("answer", "")).upper()[:1]
        instruction = _build_instruction(row)
        prompt = format_teacher_prompt(instruction, row["language"], gold)
        prompts.append(prompt)

    all_messages = [[{"role": "user", "content": p}] for p in prompts]
    LOGGER.info("Running teacher inference on %d prompts...", len(all_messages))
    BATCH_SIZE = 256
    all_raw_outputs = []

    for i in range(0, len(all_messages), BATCH_SIZE):
        batch = all_messages[i: i + BATCH_SIZE]
        LOGGER.info(
            "Batch %d / %d",
            i // BATCH_SIZE + 1,
            (len(all_messages) + BATCH_SIZE - 1) // BATCH_SIZE,
        )
        batch_outputs = prompt_vllm(
            teacher,
            tokenizer,
            batch,
            max_new_tokens=2048,
        )
        all_raw_outputs.extend(batch_outputs)

    output_path = Path(args.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped_missing_answer   = 0
    skipped_missing_reasoning = 0
    wrong_language_count     = 0
    wrong_answer_count       = 0   
    correct_answer_count     = 0

    with output_path.open("w", encoding="utf-8") as fp:
        for row, raw in zip(train_rows, all_raw_outputs):

            reasoning_match = REASONING_BLOCK_RE.search(raw)
            reasoning = reasoning_match.group(1).strip() if reasoning_match else raw.strip()

            answer_match = ANSWER_RE.search(raw)
            final_answer = answer_match.group(1).upper() if answer_match else ""

            parsed = {
                "raw_generation": raw,
                "reasoning": reasoning,
                "final_answer": final_answer,
            }

            valid, reason = is_valid_generation(parsed)
            if not valid:
                LOGGER.debug("Skipping row: %s", reason)
                if reason == "missing_answer":
                    skipped_missing_answer += 1
                else:
                    skipped_missing_reasoning += 1
                continue

            gold = str(row.get("answer", "")).upper()[:1]

    
            if final_answer == gold:
                correct_answer_count += 1
            else:
                wrong_answer_count += 1
                LOGGER.debug(
                    "Unexpected wrong answer (got %s, expected %s) — keeping anyway",
                    final_answer, gold,
                )

            detected_lang = detect_language(reasoning)
            if detected_lang != row["language"]:
                LOGGER.debug(
                    "Language mismatch: expected %s, detected %s",
                    row["language"], detected_lang,
                )
                wrong_language_count += 1

            record = {
                "question": _build_instruction(row),
                "final_answer": final_answer,
                "gold_answer": gold,
                "language": row["language"],
                "subject": row.get("subject"),
                "teacher_generation": raw,
            }
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")
            written += 1

    total_skipped = skipped_missing_answer + skipped_missing_reasoning
    total_attempted = written + total_skipped
    LOGGER.info("── Train corpus stats ──────────────────────────────")
    LOGGER.info("  Total attempted          : %d", total_attempted)
    LOGGER.info("  Saved                    : %d rows to %s", written, output_path)
    LOGGER.info("  Skipped (missing answer) : %d", skipped_missing_answer)
    LOGGER.info("  Skipped (missing reason) : %d", skipped_missing_reasoning)
    LOGGER.info("  ── Of saved rows ───────────────────────────────")
    LOGGER.info("  Correct answer           : %d / %d (%.1f%%)",
                correct_answer_count, written,
                100 * correct_answer_count / written if written else 0)
    LOGGER.info("  Wrong answer (kept)      : %d / %d (%.1f%%)",
                wrong_answer_count, written,
                100 * wrong_answer_count / written if written else 0)
    LOGGER.info("  Wrong language (kept)    : %d / %d (%.1f%%)",
                wrong_language_count, written,
                100 * wrong_language_count / written if written else 0)
    LOGGER.info("────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()