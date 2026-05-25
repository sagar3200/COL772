
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from collections import defaultdict

from tqdm.auto import tqdm

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

LOGGER = logging.getLogger(__name__)

LANGUAGES = ["en", "hindi", "bengali", "kannada", "tamil"]

ANSWER_TAG_RE = re.compile(r"####\s*ANSWER\s*:\s*([A-J])", re.IGNORECASE)
LAST_LINE_LETTER_RE = re.compile(r"\b([A-J])\b", re.IGNORECASE)



def setup_logger(level: str) -> None:
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        force=True,
    )



def format_prompt(instruction: str) -> str:
    return f"""Solve this multiple-choice question.

Rules:

1. Think step-by-step in the english language.
2. STRICT FORMAT (MUST FOLLOW):
<reasoning>Step by Step reasoning here</reasoning>

3. Keep reasoning concise
4. Reasoning MUST be inside <reasoning> tags
5. Always output step-by-step reasoning
6. Do NOT write anything outside these tags except final answer
7. Final answer format: #### ANSWER: X

Question:
{instruction}

8. Output only one of the options
9. MUST pick from given options
"""


def build_instruction(row):
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    options_text = "\n".join(
        f"({letters[i]}) {opt}" for i, opt in enumerate(row["options"])
    )
    return f"{row['question']}\n\n{options_text}"



def extract_answer(text: str) -> str:
    match = ANSWER_TAG_RE.search(text)
    if match:
        return match.group(1).upper()

    matches = LAST_LINE_LETTER_RE.findall(text)
    return matches[-1].upper() if matches else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="vLLM inference + evaluation")

    parser.add_argument("--base_model", required=True)
    parser.add_argument("--adapter_path", default="")
    parser.add_argument("--test_data", required=True)
    parser.add_argument("--output_predictions", required=True)
    parser.add_argument("--report_file", required=True)

    parser.add_argument("--max_new_tokens", type=int, default=2048)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)

    parser.add_argument(
        "--log_level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )

    return parser.parse_args()

def main():
    args = parse_args()
    setup_logger(args.log_level)

    model_name = Path(args.base_model).name
    args.output_predictions = f"predictions_{model_name}.jsonl"
    args.report_file = f"metrics_{model_name}.txt"

    data = []
    with open(args.test_data, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))

    LOGGER.info(f"Loaded {len(data)} samples")

    prompts = [
        format_prompt(build_instruction(item))
        for item in data
    ]


    LOGGER.info("Loading vLLM model...")

    llm = LLM(
        model=args.base_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=bool(args.adapter_path),
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=args.max_new_tokens,
    )

    LOGGER.info("Running inference...")

    if args.adapter_path:
        lora_request = LoRARequest( 
        lora_name="adapter",
        lora_int_id = 1,
        lora_path=args.adapter_path
        )
        outputs = llm.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )
    else:
        outputs = llm.generate(prompts, sampling_params)

    decoded = [out.outputs[0].text for out in outputs]

    predictions = []
    total = 0
    correct = 0

    lang_correct = defaultdict(int)
    lang_total = defaultdict(int)

    for item, gen_text in zip(data, decoded):
        pred = extract_answer(gen_text)

        gold = item.get("answer").upper()

        is_correct = pred == gold

        predictions.append({
            "language": item.get("language"),
            "subject": item.get("subject"),
            "question": item.get("question"),
            "gold_answer": gold,
            "predicted_answer": pred,
            "generation": gen_text,
        })

        total += 1
        if is_correct:
            correct += 1

        lang = item.get("language", "unknown")
        lang_total[lang] += 1
        if is_correct:
            lang_correct[lang] += 1


    with open(args.output_predictions, "w", encoding="utf-8") as f:
        for p in predictions:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")


    overall_acc = correct / total if total else 0

    report_lines = [f"Overall ACCURACY: {overall_acc * 100:.2f}"]

    for lang in LANGUAGES:
        if lang_total[lang] > 0:
            acc = lang_correct[lang] / lang_total[lang]
            report_lines.append(f"{lang.upper()} ACCURACY: {acc * 100:.2f}")

    report = "\n".join(report_lines)

    with open(args.report_file, "w") as f:
        f.write(report)

    LOGGER.info("\n" + report)

if __name__ == "__main__":
    main()