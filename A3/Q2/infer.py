import copy
import json
import re
import sys
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if len(sys.argv) < 4:
    print("Usage: python infer.py <lang> <test_file_path> <output_dir>")
    sys.exit(1)

LANG       = sys.argv[1].lower()
TEST_PATH  = Path(sys.argv[2]).expanduser().absolute()
OUTPUT_DIR = Path(sys.argv[3]).expanduser().absolute()
SFT_DIR    = OUTPUT_DIR / "sft_adapter"

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 8    
MAX_TOKENS = 20    
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nTask 2 Inference | lang={LANG} | device={DEVICE}")
print(f"  SFT adapter : {SFT_DIR}")


def load_jsonl(path):
    data = []
    if not Path(path).exists():
        print(f"  ERROR: test file not found → {path}")
        sys.exit(1)
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:    data.append(json.loads(line))
                except: pass
    if data:
        return data
    with open(path, encoding="utf-8") as f:
        buffer, brace_count = "", 0
        for line in f:
            line = line.strip()
            if not line: continue
            brace_count += line.count("{") - line.count("}")
            buffer      += line + " "
            if brace_count == 0 and buffer.strip():
                cleaned = buffer.strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    try:    data.append(json.loads(cleaned))
                    except: pass
                buffer = ""
    return data


def safe_load_map(path):
    if Path(path).exists():
        return json.load(open(path, encoding="utf-8"))
    return {}


def clean_sent(text):
    return text.replace("{", "").replace("}", "")


def build_prompt(sent, em1, em2, lang, tokenizer, max_length=512):
    prefix = (f"[{lang.upper()}] Extract the relation between the entities.\n"
              f"Sentence: ")
    suffix = (f"\nEntity 1: {em1}"
              f"\nEntity 2: {em2}"
              f"\nRelation:")

    prefix_len = len(tokenizer(prefix, add_special_tokens=False)["input_ids"])
    suffix_len = len(tokenizer(suffix, add_special_tokens=False)["input_ids"])
    label_len  = 15
    buffer     = 5

    sent_budget = max(20, max_length - prefix_len - suffix_len - label_len - buffer)
    sent_ids    = tokenizer(
        sent, add_special_tokens=False,
        max_length=sent_budget, truncation=True,
    )["input_ids"]
    sent_trunc = tokenizer.decode(sent_ids, skip_special_tokens=True)

    return prefix + sent_trunc + suffix

def post_process_label(generated, valid_labels, fallback="NA"):
    text = generated.strip()

    g = text.lower()
    for label in valid_labels:
        if g == label.lower():
            return label

    try:
        match = re.search(r'\{[^}]+\}', text)
        if match:
            label = json.loads(match.group()).get("label", "").strip()
            if label in valid_labels:
                return label
    except:
        pass

    for label in sorted(valid_labels, key=len, reverse=True):
        if label.lower() in g:
            return label

    match = re.search(r'/[\w/]+', text)
    if match:
        for label in valid_labels:
            if match.group() in label:
                return label

    return fallback

def english_to_lang(label_en, lang, hi_map, kn_map, or_map, tcy_map):
    if lang == "en":  return label_en
    if lang == "hi":  return hi_map.get(label_en, label_en)
    if lang == "kn":  return kn_map.get(label_en, label_en)
    if lang == "or":  return or_map.get(label_en, label_en)
    if lang == "tcy": return tcy_map.get(label_en, label_en)
    return label_en

print("\nLoading artifacts ...")

if not SFT_DIR.exists():
    print(f"  ERROR: sft_adapter not found at {SFT_DIR}")
    print(f"  Make sure training completed and output_dir is correct.")
    sys.exit(1)

valid_labels = json.load(open(SFT_DIR / "valid_labels.json", encoding="utf-8"))
print(f"  Valid labels: {len(valid_labels)}")

hi_map  = safe_load_map(SFT_DIR / "hi_map.json")
kn_map  = safe_load_map(SFT_DIR / "kn_map.json")
or_map  = safe_load_map(SFT_DIR / "or_map.json")
tcy_map = safe_load_map(SFT_DIR / "tcy_map.json")

tokenizer = AutoTokenizer.from_pretrained(str(SFT_DIR))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"
print(f"  Tokenizer vocab: {len(tokenizer)}")

print(f"  Loading base model {MODEL_NAME} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
base_model.resize_token_embeddings(len(tokenizer))

print(f"  Applying best SFT adapter from {SFT_DIR} ...")
model = PeftModel.from_pretrained(base_model, str(SFT_DIR))
model.to(DEVICE)
model.eval()
print(f"  Model ready on {DEVICE}")

print(f"\nLoading test file: {TEST_PATH} ...")
test_data = load_jsonl(TEST_PATH)
print(f"  {len(test_data)} entries")

flat_prompts = []
flat_index   = []  

for entry_idx, entry in enumerate(test_data):
    sent = clean_sent(entry["sentText"])
    for mention_idx, mention in enumerate(entry["relationMentions"]):
        em1    = mention["em1Text"]
        em2    = mention["em2Text"]
        prompt = build_prompt(sent, em1, em2, LANG, tokenizer)
        flat_prompts.append(prompt)
        flat_index.append((entry_idx, mention_idx))

print(f"  {len(flat_prompts)} entity pairs to predict")


print("\nRunning generation ...")
all_generated = []

with torch.no_grad():
    for start in tqdm(range(0, len(flat_prompts), BATCH_SIZE), desc="Batches"):
        batch_prompts = flat_prompts[start: start + BATCH_SIZE]

        inputs = tokenizer(
            batch_prompts,
            return_tensors   = "pt",
            padding          = True,   
            truncation       = True,
            max_length       = 512,
        ).to(DEVICE)

        prompt_len = inputs["input_ids"].shape[1]

        outputs = model.generate(
            **inputs,
            max_new_tokens = MAX_TOKENS,
            do_sample      = False,    
            pad_token_id   = tokenizer.pad_token_id,
            eos_token_id   = tokenizer.eos_token_id,
        )

        for i in range(outputs.shape[0]):
            new_tokens = outputs[i][prompt_len:]
            generated  = tokenizer.decode(new_tokens, skip_special_tokens=True)
            all_generated.append(generated)

out_entries = copy.deepcopy(test_data)

for (entry_idx, mention_idx), generated in zip(flat_index, all_generated):
    label_en  = post_process_label(generated, valid_labels)

    label_out = english_to_lang(label_en, LANG, hi_map, kn_map, or_map, tcy_map)

    out_entries[entry_idx]["relationMentions"][mention_idx]["label"] = label_out


out_path = OUTPUT_DIR / f"Q2_{LANG}.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for entry in out_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nSaved → {out_path}  ({len(out_entries)} entries)")