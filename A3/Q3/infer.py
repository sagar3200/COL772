import os
import json
import re
import argparse
import random
from pathlib import Path
from tqdm import tqdm

import faiss
from sentence_transformers import SentenceTransformer
from vllm import LLM, SamplingParams


_TA_MODEL = (
    "/home/scai/msr/aiy247541/scratch/"
    "models--meta-llama--Llama-3.1-8B-Instruct/snapshots/"
    "0e9e39f249a16976918f6564b8830bc894c89659"
)
MODEL_PATH = (
    _TA_MODEL if os.path.exists(_TA_MODEL)
    else "meta-llama/Meta-Llama-3.1-8B-Instruct"
)

_TA_SENT = (
    "/home/scai/msr/aiy247541/scratch/"
    "models--sentence-transformers--paraphrase-multilingual-MiniLM-L12-v2/"
    "snapshots/e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
)
SENTENCE_MODEL_PATH = (
    _TA_SENT if os.path.exists(_TA_SENT)
    else "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)

BASE_DIR = Path(__file__).resolve().parent.parent 

EN_PATH  = BASE_DIR / "en_sft_dataset" / "train.jsonl"
HI_PATH  = BASE_DIR / "sft_dataset"    / "hi_train.jsonl"
KN_PATH  = BASE_DIR / "sft_dataset"    / "kn_train.jsonl"
OR_PATH  = BASE_DIR / "sft_dataset"    / "or_train.jsonl"
TCY_PATH = BASE_DIR / "sft_dataset"    / "tcy_train.jsonl"

HI_MAP_PATH  = BASE_DIR / "sft_dataset" / "hi_map.json"
KN_MAP_PATH  = BASE_DIR / "sft_dataset" / "kn_map.json"
OR_MAP_PATH  = BASE_DIR / "sft_dataset" / "or_map.json"
TCY_MAP_PATH = BASE_DIR / "sft_dataset" / "tcy_map.json"


def load_jsonl(path):
    path = Path(path)
    if not path.exists():
        print(f"  WARNING: {path} not found")
        return []
    data    = []
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8") as f:
        content = f.read().strip()
    pos = 0
    while pos < len(content):
        while pos < len(content) and content[pos] in " \t\n\r":
            pos += 1
        if pos >= len(content):
            break
        try:
            obj, end = decoder.raw_decode(content, pos)
            data.append(obj)
            pos = end
        except Exception:
            break
    return data


def load_map(path: Path) -> dict:
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    return {}

def build_label_vocab() -> set:
    en_data = load_jsonl(EN_PATH)
    labels  = set()
    for item in en_data:
        for rel in item.get("relationMentions", []):
            lab = rel.get("label", "").strip()
            if lab:
                labels.add(lab)
    labels.add("NA")
    print(f"  Label vocab: {len(labels)} labels")
    return labels

def get_lang_map(lang: str) -> dict:
    maps = {"hi": HI_MAP_PATH, "kn": KN_MAP_PATH,
            "or": OR_MAP_PATH, "tcy": TCY_MAP_PATH}
    if lang == "en" or lang not in maps:
        return {}
    return load_map(maps[lang])


def get_rev_map(lang: str) -> dict:
    return {v: k for k, v in get_lang_map(lang).items()}


def build_pool(lang: str) -> list:

    pool = []

    for item in load_jsonl(EN_PATH):
        for rel in item.get("relationMentions", []):
            pool.append({
                "sentText": item["sentText"],
                "em1Text":  rel["em1Text"],
                "em2Text":  rel["em2Text"],
                "label":    rel["label"],
            })
    if len(pool) > 5000:
        pool = random.sample(pool, 5000)
    print(f"  EN pool: {len(pool)} examples")

    lang_paths = {
        "hi": HI_PATH, "kn": KN_PATH,
        "or": OR_PATH, "tcy": TCY_PATH,
    }
    if lang in lang_paths:
        rev_map = get_rev_map(lang)
        added   = 0
        for item in load_jsonl(lang_paths[lang]):
            sent = item["sentText"].replace("{", "").replace("}", "")
            for rel in item.get("relationMentions", []):
                raw   = rel.get("label", "NA")
                en_lb = rev_map.get(raw, raw)  
                pool.append({
                    "sentText": sent,
                    "em1Text":  rel["em1Text"],
                    "em2Text":  rel["em2Text"],
                    "label":    en_lb,
                })
                added += 1
        print(f"  {lang.upper()} pool: +{added} examples")

    print(f"  Total pool: {len(pool)} examples")
    return pool


class FAISSRetriever:

    def __init__(self, pool: list):
        self.pool  = pool
        print(f"  Loading sentence model from local cache ...")
        self.model = SentenceTransformer(SENTENCE_MODEL_PATH)

        texts = [
            f"{x['sentText']} {x['em1Text']} {x['em2Text']}"
            for x in pool
        ]
        print(f"  Encoding {len(texts)} pool examples ...")
        emb = self.model.encode(
            texts,
            batch_size=256,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        ).astype("float32")

        dim        = emb.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(emb)
        print(f"  FAISS index: {self.index.ntotal} vectors (dim={dim})")

    def retrieve(self, query: str, k: int = 5) -> list:
        vec = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype("float32")
        _, idx = self.index.search(vec, k)
        return [self.pool[i] for i in idx[0] if i < len(self.pool)]

def truncate_text(text: str, max_chars: int = 300) -> str:
    return text[:max_chars] + "..." if len(text) > max_chars else text


def build_icl_prompt(sent: str, e1: str, e2: str,
                     examples: list) -> str:
    prompt = (
        "You are an expert in relation extraction.\n"
        "Given a sentence and two entities, identify the relationship.\n"
        'Respond ONLY with JSON: {"label": "<relation>"}.\n'
        "Use NA if no clear relation exists.\n\n"
    )
    for i, ex in enumerate(examples[:3]):   # max 3 examples
        prompt += (
            f"Example {i+1}:\n"
            f"Sentence: {truncate_text(ex['sentText'], 200)}\n"
            f"Entity1: {ex['em1Text']}\n"
            f"Entity2: {ex['em2Text']}\n"
            f'Answer: {{"label": "{ex["label"]}"}}\n\n'
        )
    prompt += (
        f"Now extract:\n"
        f"Sentence: {truncate_text(sent, 300)}\n"
        f"Entity1: {e1}\n"
        f"Entity2: {e2}\n"
        f"Answer:"
    )
    return prompt


def parse_label(text: str, valid_labels: set) -> str:
    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            label = json.loads(match.group()).get("label", "").strip()
            if label in valid_labels:
                return label
    except Exception:
        pass
    for label in sorted(valid_labels, key=len, reverse=True):
        if label != "NA" and label in text:
            return label
    return "NA"


def reconstruct_output(test_data: list, pred_map: dict,
                        lang_map: dict) -> list:
    output = []
    for i, item in enumerate(test_data):
        out = {
            "articleId":        item.get("articleId", ""),
            "sentId":           item.get("sentId",    ""),
            "sentText":         item["sentText"],
            "relationMentions": [],
        }
        for j, rel in enumerate(item.get("relationMentions", [])):
            pred = pred_map.get((i, j), "NA")
            if lang_map and pred in lang_map:
                pred = lang_map[pred]
            out["relationMentions"].append({
                "em1Text": rel["em1Text"],
                "em2Text": rel["em2Text"],
                "label":   pred,
            })
        output.append(out)
    return output



def main(args):
    random.seed(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"\n{'='*55}")
    print(f"  Task 3 — ICL Inference")
    print(f"  Language  : {args.lang}")
    print(f"  Test file : {args.test_file}")
    print(f"  Output dir: {args.output_dir}")
    print(f"{'='*55}\n")

    print("Building label vocab ...")
    valid_labels = build_label_vocab()

    lang_map = get_lang_map(args.lang)

    print("\nBuilding retrieval pool ...")
    pool = build_pool(args.lang)

    print("\nBuilding FAISS retriever ...")
    retriever = FAISSRetriever(pool)

    print("\nLoading LLM ...")
    llm = LLM(
        model                  = MODEL_PATH,
        dtype                  = "float16",
        max_model_len          = 4096,
        gpu_memory_utilization = 0.75,
        disable_log_stats      = True,
    )
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    print("  LLM ready\n")

    print("Loading test data ...")
    test_data = load_jsonl(args.test_file)
    print(f"  {len(test_data)} records\n")

    print("Building prompts ...")
    prompts, meta = [], []
    for i, item in enumerate(tqdm(test_data, desc="Prompts")):
        sent = item["sentText"].replace("{", "").replace("}", "")
        for j, rel in enumerate(item.get("relationMentions", [])):
            e1  = rel["em1Text"]
            e2  = rel["em2Text"]
            q   = f"{sent} {e1} {e2}"
            ex  = retriever.retrieve(q, k=3)
            prompts.append(build_icl_prompt(sent, e1, e2, ex))
            meta.append((i, j))
    print(f"  {len(prompts)} prompts\n")

    MAX_PROMPT_CHARS = 12000   
    safe_prompts = []
    safe_meta    = []
    skipped      = 0
    for p, m in zip(prompts, meta):
        if len(p) > MAX_PROMPT_CHARS:
            p = p[:MAX_PROMPT_CHARS]
            skipped += 1
        safe_prompts.append(p)
        safe_meta.append(m)
    if skipped:
        print(f"  {skipped} prompts truncated to fit context window")

    print("Running inference ...")
    pred_map = {}
    BATCH    = 16
    for i in tqdm(range(0, len(safe_prompts), BATCH), desc="Batches"):
        batch = safe_prompts[i:i+BATCH]
        try:
            outs = llm.generate(batch, sampling_params)
            for j, out in enumerate(outs):
                pred_map[safe_meta[i+j]] = parse_label(
                    out.outputs[0].text, valid_labels
                )
        except Exception as e:
            print(f"  WARNING: batch {i//BATCH} failed ({e}) — marking NA")
            for j in range(len(batch)):
                if i+j < len(safe_meta):
                    pred_map[safe_meta[i+j]] = "NA"

    result   = reconstruct_output(test_data, pred_map, lang_map)
    out_path = os.path.join(args.output_dir, f"Q3_{args.lang}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in result:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\n  ✓ Saved {len(result)} predictions → {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lang",       required=True,
                        choices=["en", "hi", "kn", "or", "tcy"])
    parser.add_argument("--test_file",  required=True)
    parser.add_argument("--output_dir", default="output_task3")
    args = parser.parse_args()

    if not Path(args.test_file).exists():
        print(f"ERROR: test file not found → {args.test_file}")
        exit(1)

    main(args)