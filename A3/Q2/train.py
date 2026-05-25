
import json
import os
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Free memory: {torch.cuda.mem_get_info(0)[0] / 1024**3:.1f} GB")



BASE_DIR     = Path(__file__).resolve().parent.parent 

EN_PATH      = BASE_DIR / "en_sft_dataset" / "train.jsonl"
HI_PATH      = BASE_DIR / "sft_dataset"    / "hi_train.jsonl"
KN_PATH      = BASE_DIR / "sft_dataset"    / "kn_train.jsonl"
OR_PATH      = BASE_DIR / "sft_dataset"    / "or_train.jsonl"
_TCY_TRAIN   = BASE_DIR / "sft_dataset"    / "tcy_train.jsonl"
_TCY_VAL     = BASE_DIR / "sft_dataset"    / "tcy_val.jsonl"
TCY_PATH     = _TCY_TRAIN if _TCY_TRAIN.exists() else _TCY_VAL

HI_MAP_PATH  = BASE_DIR / "sft_dataset"    / "hi_map.json"
KN_MAP_PATH  = BASE_DIR / "sft_dataset"    / "kn_map.json"
OR_MAP_PATH  = BASE_DIR / "sft_dataset"    / "or_map.json"
TCY_MAP_PATH = BASE_DIR / "sft_dataset"    / "tcy_map.json"

OUTPUT_DIR   = Path(sys.argv[1]) if len(sys.argv) > 1 \
               else BASE_DIR / "output_task2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CPT_DIR = OUTPUT_DIR / "cpt_adapter"
SFT_DIR = OUTPUT_DIR / "sft_adapter"
CPT_DIR.mkdir(parents=True, exist_ok=True)
SFT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME   = "Qwen/Qwen2.5-1.5B"
SEED         = 42
HI_KN_MULT   = 8

LANG_MAX_LEN = {
    "en":  200,   
    "hi":  400,
    "kn":  512,
    "or":  512,
    "tcy": 512,
}
CPT_MAX_LEN  = 512
SFT_MAX_LEN  = 512  

BATCH_SIZE   = 8     
GRAD_ACCUM   = 4     

CPT_EPOCHS   = 1
CPT_LR       = 1e-4

SFT_EPOCHS   = 5
SFT_LR       = 2e-4

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROP    = 0.05

WIKI_LIMITS  = {"hi": 10000, "kn": 10000, "or": 10000, "tcy": None}
TULU_REPEAT  = 5

DEBUG        = False

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)



def load_jsonl_singleline(path: Path):
    data = []
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return data
    with open(path, encoding="utf-8") as f:
        for line in tqdm(f, desc=f"Loading {path.name}", unit=" lines"):
            line = line.strip()
            if line:
                try:    data.append(json.loads(line))
                except: pass
    return data


def load_jsonl_multiline(path: Path):
    data = []
    if not path.exists():
        print(f"  WARNING: {path.name} not found")
        return data
    with open(path, encoding="utf-8") as f:
        buffer, brace_count = "", 0
        for line in tqdm(f, desc=f"Loading {path.name}", unit=" lines"):
            line = line.strip()
            if not line:
                continue
            brace_count += line.count("{") - line.count("}")
            buffer      += line + " "
            if brace_count == 0 and buffer.strip():
                cleaned = buffer.strip()
                if cleaned.startswith("{") and cleaned.endswith("}"):
                    try:    data.append(json.loads(cleaned))
                    except: pass
                buffer = ""
    return data


def safe_load_map(path: Path):
    if path.exists():
        return json.load(open(path, encoding="utf-8"))
    print(f"  WARNING: map not found → {path}")
    return {}


def normalize_label(label, lang, rev_hi, rev_kn, rev_or, rev_tcy):
    if lang == "en":  return label
    if lang == "hi":  return rev_hi.get(label, "NA")
    if lang == "kn":  return rev_kn.get(label, "NA")
    if lang == "or":  return rev_or.get(label, "NA")
    if lang == "tcy": return rev_tcy.get(label, "NA")
    return "NA"


def clean_sent(text: str) -> str:
    return text.replace("{", "").replace("}", "")


def build_prompt_safe(sent, em1, em2, lang, tokenizer, max_length=512):
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

    sent_ids = tokenizer(
        sent, add_special_tokens=False,
        max_length=sent_budget, truncation=True,
    )["input_ids"]
    sent_truncated = tokenizer.decode(sent_ids, skip_special_tokens=True)

    return prefix + sent_truncated + suffix


def post_process_label(generated: str, valid_labels: list) -> str:
    g = generated.strip().lower()
    for label in valid_labels:
        if g == label.lower():
            return label
    for label in valid_labels:
        if label.lower() in g:
            return label
    return "NA"


def create_sft_samples(data, lang, rev_hi, rev_kn, rev_or, rev_tcy):
    samples = []
    for item in tqdm(data, desc=f"Processing {lang.upper()} records"):
        sent = clean_sent(item["sentText"])
        for rel in item["relationMentions"]:
            em1   = rel["em1Text"]
            em2   = rel["em2Text"]
            label = normalize_label(rel.get("label", "NA"), lang,
                                    rev_hi, rev_kn, rev_or, rev_tcy)
            samples.append({
                "sentence": sent,
                "em1": em1, "em2": em2,
                "label": label, "lang": lang,
            })
    return samples


def safe_split(samples, test_size):
    if len(samples) == 0:  return [], []
    if len(samples) < 10:  return samples, []
    return train_test_split(samples, test_size=test_size, random_state=SEED)



class WikiCPTDataset(Dataset):

    def __init__(self, texts, tokenizer, max_length=512):
        self.items     = []
        self.tokenizer = tokenizer
        for text in tqdm(texts, desc="Building CPT dataset"):
            if not text.strip():
                continue
            enc       = tokenizer(text, max_length=max_length,
                                  truncation=True)  
            input_ids = enc["input_ids"]
            self.items.append(input_ids)
        print(f"  CPT dataset: {len(self.items)} samples")

    def __len__(self):        return len(self.items)

    def __getitem__(self, i):
        return {"input_ids": self.items[i]}


class SFTDataset(Dataset):

    def __init__(self, samples, tokenizer, max_length=512, name=""):
        self.items     = []
        self.tokenizer = tokenizer
        skipped        = 0

        for s in tqdm(samples, desc=f"Building SFT {name} dataset"):
            lang_max = LANG_MAX_LEN.get(s["lang"], max_length)

            prompt     = build_prompt_safe(
                s["sentence"], s["em1"], s["em2"], s["lang"],
                tokenizer, lang_max
            )
            completion = f" {s['label']}"
            full_text  = prompt + completion

            full_ids   = tokenizer(full_text, add_special_tokens=True)["input_ids"]
            if len(full_ids) > lang_max:
                skipped += 1
                continue

            prompt_len = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])

            if prompt_len >= len(full_ids):
                skipped += 1
                continue

            self.items.append({
                "input_ids":  full_ids,
                "prompt_len": prompt_len,
            })

        print(f"  SFT {name}: {len(self.items)} kept, {skipped} skipped")

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


class DynamicPaddingCollatorCPT:

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        seqs     = [item["input_ids"] for item in batch]
        max_len  = max(len(s) for s in seqs)

        input_ids      = []
        attention_mask = []
        labels         = []

        for seq in seqs:
            pad_len = max_len - len(seq)
            padded  = seq + [self.pad_id] * pad_len
            mask    = [1]  * len(seq) + [0] * pad_len
            lbl     = seq + [-100]    * pad_len  

            input_ids.append(padded)
            attention_mask.append(mask)
            labels.append(lbl)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


class DynamicPaddingCollatorSFT:

    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, batch):
        seqs        = [item["input_ids"]  for item in batch]
        prompt_lens = [item.get("prompt_len", 0) for item in batch]
        max_len     = max(len(s) for s in seqs)

        input_ids      = []
        attention_mask = []
        labels         = []

        for seq, p_len in zip(seqs, prompt_lens):
            pad_len = max_len - len(seq)
            padded  = seq + [self.pad_id] * pad_len
            mask    = [1] * len(seq) + [0] * pad_len

            lbl = ([-100] * p_len         
                   + seq[p_len:]          
                   + [-100] * pad_len)    

            input_ids.append(padded)
            attention_mask.append(mask)
            labels.append(lbl)

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


class GenerationF1Callback(TrainerCallback):

    def __init__(self, val_samples, tokenizer, valid_labels,
                 max_length=512, n_eval=500, patience=3, batch_size=16):
        self.val_samples  = val_samples
        self.tokenizer    = tokenizer
        self.valid_labels = valid_labels
        self.max_length   = max_length
        self.n_eval       = n_eval
        self.patience     = patience
        self.batch_size   = batch_size
        self.best_f1      = 0.0
        self.no_improve   = 0
        self.best_epoch   = 0

        self._eval_samples = self._stratified_sample(val_samples, n_eval)

    def _stratified_sample(self, samples, n):
        from collections import defaultdict
        by_lang = defaultdict(list)
        for s in samples:
            by_lang[s["lang"]].append(s)
        result = []
        for lang, items in by_lang.items():
            take = min(20, len(items))
            result.extend(items[:take])
        remaining = n - len(result)
        if remaining > 0:
            rest = [s for s in samples if s not in result]
            import random
            result.extend(random.sample(rest, min(remaining, len(rest))))
        return result[:n]

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        model.eval()
        device   = next(model.parameters()).device
        samples  = self._eval_samples
        tokenizer = self.tokenizer

        original_padding_side    = tokenizer.padding_side
        tokenizer.padding_side   = "left"

        preds, golds = [], []

        with torch.no_grad():
            for start in tqdm(
                range(0, len(samples), self.batch_size),
                desc=f"  Gen F1 eval (epoch {int(state.epoch)})",
                leave=False
            ):
                batch = samples[start: start + self.batch_size]
                prompts = [
                    build_prompt_safe(
                        s["sentence"], s["em1"], s["em2"], s["lang"],
                        tokenizer, self.max_length
                    )
                    for s in batch
                ]

                inputs = tokenizer(
                    prompts,
                    return_tensors = "pt",
                    padding        = True,
                    truncation     = True,
                    max_length     = self.max_length,
                ).to(device)

                prompt_len = inputs["input_ids"].shape[1]

                out = model.generate(
                    **inputs,
                    max_new_tokens = 20,
                    do_sample      = False,
                    pad_token_id   = tokenizer.pad_token_id,
                    eos_token_id   = tokenizer.eos_token_id,
                )

                for i in range(out.shape[0]):
                    new_tokens = out[i][prompt_len:]
                    generated  = tokenizer.decode(
                        new_tokens, skip_special_tokens=True
                    )
                    preds.append(post_process_label(generated, self.valid_labels))
                    golds.append(batch[i]["label"])

        tokenizer.padding_side = original_padding_side
        model.train()

        micro = f1_score(golds, preds, average="micro", zero_division=0)
        macro = f1_score(golds, preds, average="macro", zero_division=0)

        langs = {}
        for s, p, g in zip(samples, preds, golds):
            langs.setdefault(s["lang"], {"p": [], "g": []})
            langs[s["lang"]]["p"].append(p)
            langs[s["lang"]]["g"].append(g)

        print(f"\n  ┌── Generation F1  Epoch {int(state.epoch)} ─────────────┐")
        print(f"  │  micro_F1 = {micro:.4f}   macro_F1 = {macro:.4f}        │")
        for lg, d in sorted(langs.items()):
            lf = f1_score(d["g"], d["p"], average="micro", zero_division=0)
            print(f"  │    {lg.upper():3s}  micro-F1 = {lf:.4f}  (n={len(d['g'])})    │")
        print(f"  └────────────────────────────────────────────────┘\n")

        if micro > self.best_f1:
            self.best_f1    = micro
            self.best_epoch = int(state.epoch)
            self.no_improve = 0
            print(f"  ✓ New best gen micro_F1 = {micro:.4f} (epoch {self.best_epoch})")
        else:
            self.no_improve += 1
            print(f"  ⚠ No improvement for {self.no_improve} epoch(s) "
                  f"(best={self.best_f1:.4f})")

        if self.no_improve >= self.patience:
            print("  Early stopping triggered by GenerationF1Callback")
            control.should_training_stop = True

class TqdmProgressCallback(TrainerCallback):

    def __init__(self, stage_name=""):
        self.stage_name = stage_name
        self.epoch_bar  = None
        self.step_bar   = None

    def on_train_begin(self, args, state, control, **kwargs):
        total = int(args.num_train_epochs)
        spe   = state.max_steps // total if total else state.max_steps
        print(f"\n{'='*55}")
        print(f"  Stage           : {self.stage_name}")
        print(f"  Epochs          : {total}")
        print(f"  Steps / epoch   : {spe}")
        print(f"  Total steps     : {state.max_steps}")
        print(f"  Batch size      : {args.per_device_train_batch_size}")
        print(f"{'='*55}\n")
        self.epoch_bar = tqdm(total=total, desc=f"{self.stage_name} Epochs",
                              unit="epoch", position=0, leave=True)

    def on_epoch_begin(self, args, state, control, **kwargs):
        total = int(args.num_train_epochs)
        steps = state.max_steps // total if total else state.max_steps
        self.step_bar = tqdm(total=steps,
                             desc=f"  Epoch {int(state.epoch)+1} steps",
                             unit=" steps", position=1, leave=False)

    def on_step_end(self, args, state, control, **kwargs):
        if self.step_bar is None: return
        self.step_bar.update(1)
        if state.log_history:
            last = state.log_history[-1]
            if "loss" in last:
                self.step_bar.set_postfix(
                    loss=f"{last['loss']:.4f}",
                    lr=f"{last.get('learning_rate', 0):.2e}",
                )

    def on_epoch_end(self, args, state, control, **kwargs):
        if self.step_bar  is not None: self.step_bar.close()
        if self.epoch_bar is not None: self.epoch_bar.update(1)

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None: self.epoch_bar.close()
        print(f"\n  {self.stage_name} training finished.\n")


class BestSFTSaverCallback(TrainerCallback):

    def __init__(self, sft_dir, tokenizer, label2id, id2label,
                 valid_labels, gen_f1_cb,
                 hi_map_path, kn_map_path, or_map_path, tcy_map_path):
        self.sft_dir      = Path(sft_dir)
        self.tokenizer    = tokenizer
        self.label2id     = label2id
        self.id2label     = id2label
        self.valid_labels = valid_labels
        self.gen_f1_cb    = gen_f1_cb  
        self.hi_map_path  = hi_map_path
        self.kn_map_path  = kn_map_path
        self.or_map_path  = or_map_path
        self.tcy_map_path = tcy_map_path
        self._last_saved_f1 = -1.0

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        current_best = self.gen_f1_cb.best_f1
        if current_best <= self._last_saved_f1:
            return

        self._last_saved_f1 = current_best
        print(f"\n  ✓ Saving SFT artifacts (gen F1={current_best:.4f}) ...")

        raw_model = model
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module

        raw_model.save_pretrained(str(self.sft_dir))
        self.tokenizer.save_pretrained(str(self.sft_dir))

        json.dump(self.label2id,
                  open(self.sft_dir / "label2id.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump({str(k): v for k, v in self.id2label.items()},
                  open(self.sft_dir / "id2label.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump(self.valid_labels,
                  open(self.sft_dir / "valid_labels.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        for src, fname in [
            (self.hi_map_path,  "hi_map.json"),
            (self.kn_map_path,  "kn_map.json"),
            (self.or_map_path,  "or_map.json"),
            (self.tcy_map_path, "tcy_map.json"),
        ]:
            if Path(src).exists():
                shutil.copy(src, self.sft_dir / fname)

        print(f"  ✓ SFT artifacts saved → {self.sft_dir}")


print("\nLoading RE data ...")
en_data  = load_jsonl_singleline(EN_PATH)
hi_data  = load_jsonl_multiline(HI_PATH)
kn_data  = load_jsonl_multiline(KN_PATH)
or_data  = load_jsonl_multiline(OR_PATH)
tcy_data = load_jsonl_multiline(TCY_PATH)

hi_map  = safe_load_map(HI_MAP_PATH)
kn_map  = safe_load_map(KN_MAP_PATH)
or_map  = safe_load_map(OR_MAP_PATH)
tcy_map = safe_load_map(TCY_MAP_PATH)

rev_hi  = {v: k for k, v in hi_map.items()}
rev_kn  = {v: k for k, v in kn_map.items()}
rev_or  = {v: k for k, v in or_map.items()}
rev_tcy = {v: k for k, v in tcy_map.items()}

print(f"  EN={len(en_data)} HI={len(hi_data)} KN={len(kn_data)} "
      f"OR={len(or_data)} TCY={len(tcy_data)}")


print("\nBuilding SFT splits ...")
en_s  = create_sft_samples(en_data,  "en",  rev_hi, rev_kn, rev_or, rev_tcy)
hi_s  = create_sft_samples(hi_data,  "hi",  rev_hi, rev_kn, rev_or, rev_tcy)
kn_s  = create_sft_samples(kn_data,  "kn",  rev_hi, rev_kn, rev_or, rev_tcy)
or_s  = create_sft_samples(or_data,  "or",  rev_hi, rev_kn, rev_or, rev_tcy)
tcy_s = create_sft_samples(tcy_data, "tcy", rev_hi, rev_kn, rev_or, rev_tcy)

en_tr,  en_val  = safe_split(en_s,  0.10)
hi_tr,  hi_val  = safe_split(hi_s,  0.15)
kn_tr,  kn_val  = safe_split(kn_s,  0.15)
or_tr,  or_val  = safe_split(or_s,  0.15)
tcy_tr, tcy_val = safe_split(tcy_s, 0.15)

train_samples = (en_tr
                 + hi_tr  * HI_KN_MULT
                 + kn_tr  * HI_KN_MULT
                 + or_tr  * 5
                 + tcy_tr * 5)
val_samples   = en_val + hi_val + kn_val + or_val + tcy_val

if len(val_samples) == 0:
    print("WARNING: val empty — using 5% of en_tr")
    en_tr, val_samples = train_test_split(en_tr, test_size=0.05, random_state=SEED)
    train_samples = (en_tr + hi_tr * HI_KN_MULT + kn_tr * HI_KN_MULT
                     + or_tr * 15 + tcy_tr * 15)

random.shuffle(train_samples)
random.shuffle(val_samples)
print(f"  train : {len(train_samples)}  val : {len(val_samples)}")

if DEBUG:
    train_samples = train_samples[:300]
    val_samples   = val_samples[:60]
    SFT_EPOCHS    = 2
    CPT_EPOCHS    = 1
    WIKI_LIMITS   = {"hi": 100, "kn": 100, "or": 100, "tcy": 50}
    print("⚠  DEBUG MODE ON")


all_labels   = sorted(set(s["label"] for s in train_samples + val_samples))
label2id     = {l: i for i, l in enumerate(all_labels)}
id2label     = {i: l for l, i in label2id.items()}
VALID_LABELS = all_labels
print(f"\n  {len(all_labels)} distinct relation labels")


print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"
print(f"  Vocab size : {len(tokenizer)}")


print("\nLoading base model ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
base_model.resize_token_embeddings(len(tokenizer))

base_model.lm_head.weight = nn.Parameter(
    base_model.lm_head.weight.detach().clone()
)

lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout   = LORA_DROP,
    bias           = "none",
    task_type      = "CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)
model.print_trainable_parameters()


print("\n" + "="*55)
print("  STAGE 1: Continued Pre-Training (Wikipedia)")
print("="*55)

def load_wiki(lang_code, max_samples=None, min_len=100):
    print(f"  Loading {lang_code} Wikipedia ...")
    ds    = load_dataset("wikimedia/wikipedia",
                         f"20231101.{lang_code}",
                         split="train")
    texts = []
    skip  = 0
    for art in tqdm(ds, desc=f"  {lang_code}"):
        t = art["text"].strip()
        if len(t) < min_len:
            skip += 1
            continue
        texts.append(t[:2000])
        if max_samples and len(texts) >= max_samples:
            break
    print(f"  {lang_code}: {len(texts)} articles, {skip} stubs skipped")
    return texts

wiki_hi  = load_wiki("hi",  WIKI_LIMITS["hi"])
wiki_kn  = load_wiki("kn",  WIKI_LIMITS["kn"])
wiki_or  = load_wiki("or",  WIKI_LIMITS["or"])
wiki_tcy = load_wiki("tcy", WIKI_LIMITS["tcy"])

all_wiki = wiki_hi + wiki_kn + wiki_or + wiki_tcy * TULU_REPEAT
random.shuffle(all_wiki)
print(f"\n  Total CPT samples : {len(all_wiki)}")

cpt_dataset  = WikiCPTDataset(all_wiki, tokenizer, CPT_MAX_LEN)
cpt_collator = DynamicPaddingCollatorCPT(tokenizer)

cpt_args = TrainingArguments(
    output_dir                  = str(CPT_DIR),
    num_train_epochs            = CPT_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    gradient_accumulation_steps = GRAD_ACCUM,
    learning_rate               = CPT_LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 100,
    weight_decay                = 0.01,
    eval_strategy               = "no",
    save_strategy               = "epoch",
    fp16                        = torch.cuda.is_available(),
    bf16                        = False,
    logging_steps               = 200,
    report_to                   = "none",
    seed                        = SEED,
    dataloader_num_workers      = 0,
    dataloader_pin_memory       = False,
)

cpt_trainer = Trainer(
    model            = model,
    args             = cpt_args,
    train_dataset    = cpt_dataset,
    data_collator    = cpt_collator,   # ← dynamic padding
    callbacks        = [TqdmProgressCallback("CPT")],
)

cpt_trainer.train()

print("\nSaving CPT adapter ...")
model.save_pretrained(str(CPT_DIR))
tokenizer.save_pretrained(str(CPT_DIR))
print(f"  CPT adapter → {CPT_DIR}")


print("\n" + "="*55)
print("  STAGE 2: Supervised Fine-Tuning (RE)")
print("="*55)

sft_train    = SFTDataset(train_samples, tokenizer, SFT_MAX_LEN, "train")
sft_val      = SFTDataset(val_samples,   tokenizer, SFT_MAX_LEN, "val")
sft_collator = DynamicPaddingCollatorSFT(tokenizer)

gen_f1_cb = GenerationF1Callback(
    val_samples  = val_samples,
    tokenizer    = tokenizer,
    valid_labels = VALID_LABELS,
    max_length   = SFT_MAX_LEN,
    n_eval       = 500,     
    batch_size   = 8,     
    patience     = 3,
)

sft_args = TrainingArguments(
    output_dir                  = str(SFT_DIR),
    num_train_epochs            = SFT_EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE * 2,
    gradient_accumulation_steps = GRAD_ACCUM,       
    learning_rate               = SFT_LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 200,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "eval_loss",
    greater_is_better           = False,
    save_total_limit            = SFT_EPOCHS,      
    fp16                        = torch.cuda.is_available(),
    bf16                        = False,
    logging_steps               = 100,
    report_to                   = "none",
    seed                        = SEED,
    dataloader_num_workers      = 0,
    dataloader_pin_memory       = False,
)

best_sft_saver = BestSFTSaverCallback(
    sft_dir      = SFT_DIR,
    tokenizer    = tokenizer,
    label2id     = label2id,
    id2label     = id2label,
    valid_labels = VALID_LABELS,
    gen_f1_cb    = gen_f1_cb,
    hi_map_path  = HI_MAP_PATH,
    kn_map_path  = KN_MAP_PATH,
    or_map_path  = OR_MAP_PATH,
    tcy_map_path = TCY_MAP_PATH,
)

sft_trainer = Trainer(
    model         = model,
    args          = sft_args,
    train_dataset = sft_train,
    eval_dataset  = sft_val,
    data_collator = sft_collator,
    callbacks     = [gen_f1_cb, TqdmProgressCallback("SFT"), best_sft_saver],
)

sft_trainer.train()


print("\nSaving final artefacts ...")

model.save_pretrained(str(SFT_DIR))
tokenizer.save_pretrained(str(SFT_DIR))

json.dump(label2id,
          open(SFT_DIR / "label2id.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump({str(k): v for k, v in id2label.items()},
          open(SFT_DIR / "id2label.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump(VALID_LABELS,
          open(SFT_DIR / "valid_labels.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

for src, fname in [
    (HI_MAP_PATH,  "hi_map.json"),
    (KN_MAP_PATH,  "kn_map.json"),
    (OR_MAP_PATH,  "or_map.json"),
    (TCY_MAP_PATH, "tcy_map.json"),
]:
    if src.exists():
        shutil.copy(src, SFT_DIR / fname)
    else:
        print(f"  WARNING: {src.name} not found")

print(f"\nAll artefacts saved → {SFT_DIR}")
print(f"Files: {[p.name for p in sorted(SFT_DIR.iterdir())]}")
print(f"\nBest generation F1 = {gen_f1_cb.best_f1:.4f} "
      f"at epoch {gen_f1_cb.best_epoch}")