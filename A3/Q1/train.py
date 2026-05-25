import json
import os
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.modeling_outputs import SequenceClassifierOutput
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

HI_MAP_PATH  = BASE_DIR / "sft_dataset" / "hi_map.json"
KN_MAP_PATH  = BASE_DIR / "sft_dataset" / "kn_map.json"

OUTPUT_DIR   = (Path(sys.argv[1]).expanduser().absolute()
                if len(sys.argv) > 1
                else BASE_DIR / "output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


MODEL_NAME       = "Qwen/Qwen2.5-1.5B"
MAX_LEN          = 512
SEED             = 42
HI_KN_MULT       = 10

BATCH_SIZE       = 8    
GRAD_ACCUM       = 4    
EPOCHS           = 5
LR               = 2e-4
LORA_R           = 8
LORA_ALPHA       = 16
LORA_DROP        = 0.1

DEBUG            = False

random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def load_jsonl_singleline(path: Path):
    data = []
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


print("\nLoading data ...")
en_data = load_jsonl_singleline(EN_PATH)
hi_data = load_jsonl_multiline(HI_PATH)
kn_data = load_jsonl_multiline(KN_PATH)

hi_map  = safe_load_map(HI_MAP_PATH)
kn_map  = safe_load_map(KN_MAP_PATH)

rev_hi  = {v: k for k, v in hi_map.items()}
rev_kn  = {v: k for k, v in kn_map.items()}

print(f"  EN={len(en_data)}  HI={len(hi_data)}  KN={len(kn_data)}")


def normalize_label(label: str, lang: str) -> str:
    if lang == "en": return label
    if lang == "hi": return rev_hi.get(label, "NA")
    if lang == "kn": return rev_kn.get(label, "NA")
    return "NA"


def mark_entities(text: str, e1: str, e2: str) -> str:
    if e1 in text:
        text = text.replace(e1, f"[E1] {e1} [/E1]", 1)
    if e2 in text:
        text = text.replace(e2, f"[E2] {e2} [/E2]", 1)
    return text


def create_samples(data, lang: str):
    samples = []
    for item in tqdm(data, desc=f"Processing {lang.upper()} records"):
        sent = item["sentText"]
        for rel in item["relationMentions"]:
            em1   = rel["em1Text"]
            em2   = rel["em2Text"]
            label = normalize_label(rel.get("label", "NA"), lang)
            samples.append({
                "sentence": sent,
                "em1":      em1,
                "em2":      em2,
                "label":    label,
                "lang":     lang,
            })
    return samples


def safe_split(samples, test_size):
    if len(samples) == 0:  return [], []
    if len(samples) < 10:  return samples, []
    return train_test_split(samples, test_size=test_size, random_state=SEED)


print("\nBuilding splits ...")
en_samples = create_samples(en_data, "en")
hi_samples = create_samples(hi_data, "hi")
kn_samples = create_samples(kn_data, "kn")

en_tr, en_val = safe_split(en_samples, test_size=0.10)
hi_tr, hi_val = safe_split(hi_samples, test_size=0.15)
kn_tr, kn_val = safe_split(kn_samples, test_size=0.15)

train_samples = en_tr + hi_tr * HI_KN_MULT + kn_tr * HI_KN_MULT
val_samples   = en_val + hi_val + kn_val

if len(val_samples) == 0:
    print("WARNING: val empty — using 5% of en_tr")
    en_tr, val_samples = train_test_split(en_tr, test_size=0.05, random_state=SEED)
    train_samples = en_tr + hi_tr * HI_KN_MULT + kn_tr * HI_KN_MULT

random.shuffle(train_samples)
random.shuffle(val_samples)

print(f"  train : {len(train_samples)}  val : {len(val_samples)}")

if DEBUG:
    train_samples = train_samples[:200]
    val_samples   = val_samples[:50]
    EPOCHS        = 2
    print("⚠  DEBUG MODE ON")

all_labels = sorted(set(s["label"] for s in train_samples + val_samples))
label2id   = {l: i for i, l in enumerate(all_labels)}
id2label   = {i: l for l, i in label2id.items()}
NUM_LABELS = len(all_labels)
print(f"\n  {NUM_LABELS} distinct relation labels")

print("\nLoading tokenizer ...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.add_special_tokens({
    "additional_special_tokens": ["[E1]", "[/E1]", "[E2]", "[/E2]"]
})
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

e1_id = tokenizer.convert_tokens_to_ids("[E1]")
e2_id = tokenizer.convert_tokens_to_ids("[E2]")
print(f"  Vocab size : {len(tokenizer)}")
print(f"  [E1] id = {e1_id}  |  [E2] id = {e2_id}")


class REDataset(Dataset):
    def __init__(self, samples):
        self.items = []
        skipped    = 0
        for s in tqdm(samples, desc="Building dataset"):
            text     = f"[{s['lang'].upper()}] " + mark_entities(
                s["sentence"], s["em1"], s["em2"]
            )
            enc      = tokenizer(text, truncation=True, max_length=MAX_LEN)
            if e1_id not in enc["input_ids"] or e2_id not in enc["input_ids"]:
                skipped += 1
                continue
            self.items.append({
                "input_ids":      enc["input_ids"],
                "attention_mask": enc["attention_mask"],
                "label":          label2id[s["label"]],
                "lang":           s["lang"],  
            })
        print(f"  Dataset: {len(self.items)} kept, {skipped} skipped "
              f"(entity marker truncated)")

    def __len__(self):        return len(self.items)
    def __getitem__(self, i): return self.items[i]


class DynamicPaddingCollator:

    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len        = max(len(item["input_ids"]) for item in batch)
        input_ids      = []
        attention_mask = []
        labels         = []

        for item in batch:
            seq     = item["input_ids"]
            mask    = item["attention_mask"]
            pad_len = max_len - len(seq)

            input_ids.append(seq  + [self.pad_id] * pad_len)
            attention_mask.append(mask + [0]       * pad_len)
            labels.append(item["label"])

        return {
            "input_ids":      torch.tensor(input_ids,      dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels":         torch.tensor(labels,         dtype=torch.long),
        }


label_counts  = Counter(s["label"] for s in train_samples)
total_count   = sum(label_counts.values())
class_weights = torch.tensor(
    [total_count / (NUM_LABELS * label_counts.get(id2label[i], 1))
     for i in range(NUM_LABELS)],
    dtype=torch.float32,
)


class REClassifier(nn.Module):

    def __init__(self, encoder, num_labels, e1_id, e2_id, class_weights):
        super().__init__()
        self.encoder = encoder
        self.e1_id   = e1_id
        self.e2_id   = e2_id
        self.register_buffer("class_weights", class_weights.float())

        hidden = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def _get_entity_rep(self, hidden_states, input_ids, token_id):
        reps = []
        for i in range(hidden_states.size(0)):
            positions = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
            if len(positions) > 0:
                reps.append(hidden_states[i, positions[0].item(), :])
            else:
                mask = (input_ids[i] != tokenizer.pad_token_id)
                last = mask.nonzero(as_tuple=True)[0][-1].item()
                reps.append(hidden_states[i, last, :])
        return torch.stack(reps)

    def forward(self, input_ids, attention_mask, labels=None):
        out    = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]                         
        e1_rep = self._get_entity_rep(hidden, input_ids, self.e1_id)
        e2_rep = self._get_entity_rep(hidden, input_ids, self.e2_id)
        pair   = torch.cat([e1_rep, e2_rep], dim=-1).float()  
        logits = self.classifier(pair)                         

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(weight=self.class_weights)(logits, labels)

        return SequenceClassifierOutput(loss=loss, logits=logits)


print("\nLoading base model ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float16,
)
base_model.config.output_hidden_states = True
base_model.resize_token_embeddings(len(tokenizer))
base_model.lm_head.weight = nn.Parameter(
    base_model.lm_head.weight.detach().clone()
)

lora_config = LoraConfig(
    r              = LORA_R,
    lora_alpha     = LORA_ALPHA,
    target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout   = LORA_DROP,
    bias           = "none",
    task_type      = "FEATURE_EXTRACTION",
)
lora_model = get_peft_model(base_model, lora_config)
lora_model.print_trainable_parameters()

model     = REClassifier(lora_model, NUM_LABELS, e1_id, e2_id, class_weights)
collator  = DynamicPaddingCollator(tokenizer.pad_token_id)

train_dataset = REDataset(train_samples)
val_dataset   = REDataset(val_samples)


class TqdmProgressCallback(TrainerCallback):

    def __init__(self):
        self.epoch_bar = None
        self.step_bar  = None

    def on_train_begin(self, args, state, control, **kwargs):
        total    = int(args.num_train_epochs)
        spe      = state.max_steps // total if total else state.max_steps
        eff_batch = args.per_device_train_batch_size * args.gradient_accumulation_steps
        print(f"\n{'='*55}")
        print(f"  Epochs          : {total}")
        print(f"  Steps/epoch     : {spe}")
        print(f"  Total steps     : {state.max_steps}")
        print(f"  Batch size      : {args.per_device_train_batch_size}")
        print(f"  Grad accum      : {args.gradient_accumulation_steps}")
        print(f"  Effective batch : {eff_batch}")
        print(f"{'='*55}\n")
        self.epoch_bar = tqdm(total=total, desc="Epochs",
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

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if not metrics: return
        print(f"\n  ┌── Eval Epoch {int(state.epoch)} ──────────────┐")
        print(f"  │  loss     = {metrics.get('eval_loss', 0):.4f}          │")
        print(f"  │  micro_F1 = {metrics.get('eval_micro_f1', 0):.4f}          │")
        print(f"  │  macro_F1 = {metrics.get('eval_macro_f1', 0):.4f}          │")
        print(f"  └────────────────────────────────────────┘\n")

    def on_train_end(self, args, state, control, **kwargs):
        if self.epoch_bar is not None: self.epoch_bar.close()
        print(f"\n  Training finished.")
        print(f"  Best micro_F1   = {state.best_metric}")
        print(f"  Best checkpoint = {state.best_model_checkpoint}\n")

class BestModelSaverCallback(TrainerCallback):

    def __init__(self, output_dir, tokenizer, label2id, id2label,
                 max_len, num_labels, e1_id, e2_id,
                 hi_map_path, kn_map_path):
        self.output_dir   = Path(output_dir)
        self.tokenizer    = tokenizer
        self.label2id     = label2id
        self.id2label     = id2label
        self.max_len      = max_len
        self.num_labels   = num_labels
        self.e1_id        = e1_id
        self.e2_id        = e2_id
        self.hi_map_path  = hi_map_path
        self.kn_map_path  = kn_map_path
        self.best_f1      = 0.0

    def on_evaluate(self, args, state, control, model=None, metrics=None, **kwargs):
        if metrics is None: return

        current_f1 = metrics.get("eval_micro_f1", 0.0)
        if current_f1 <= self.best_f1:
            return

        self.best_f1 = current_f1
        print(f"\n  ✓ New best micro_F1={current_f1:.4f} — saving inference artifacts ...")

        raw_model = model
        if hasattr(raw_model, "module"):
            raw_model = raw_model.module

        lora_dir = self.output_dir / "lora_adapter"
        raw_model.encoder.save_pretrained(str(lora_dir))

        tok_dir = self.output_dir / "tokenizer"
        self.tokenizer.save_pretrained(str(tok_dir))

        torch.save(
            raw_model.classifier.state_dict(),
            self.output_dir / "classifier_head.pt",
        )

        json.dump(
            {"max_length": self.max_len, "num_labels": self.num_labels,
             "e1_id": self.e1_id, "e2_id": self.e2_id},
            open(self.output_dir / "train_config.json", "w"),
            indent=2,
        )

        json.dump(self.label2id,
                  open(self.output_dir / "label2id.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)
        json.dump({str(k): v for k, v in self.id2label.items()},
                  open(self.output_dir / "id2label.json", "w", encoding="utf-8"),
                  ensure_ascii=False, indent=2)

        for src, fname in [(self.hi_map_path, "hi_map.json"),
                           (self.kn_map_path, "kn_map.json")]:
            if Path(src).exists():
                shutil.copy(src, self.output_dir / fname)

        print(f"  ✓ Artifacts saved → {self.output_dir}")

def compute_metrics(eval_pred):
    logits, label_ids = eval_pred
    preds     = np.argmax(logits, axis=-1)
    pred_strs = [id2label[p] for p in preds]
    gold_strs = [id2label[g] for g in label_ids]

    micro = f1_score(gold_strs, pred_strs, average="micro", zero_division=0)
    macro = f1_score(gold_strs, pred_strs, average="macro", zero_division=0)

    langs = {}
    for item, p, g in zip(val_dataset.items[:len(preds)], pred_strs, gold_strs):
        lg = item["lang"]
        langs.setdefault(lg, {"p": [], "g": []})
        langs[lg]["p"].append(p)
        langs[lg]["g"].append(g)
    for lg, d in sorted(langs.items()):
        lf = f1_score(d["g"], d["p"], average="micro", zero_division=0)
        print(f"    {lg.upper()} micro-F1 = {lf:.4f}  (n={len(d['g'])})")

    return {"micro_f1": micro, "macro_f1": macro}

training_args = TrainingArguments(
    output_dir                  = str(OUTPUT_DIR),
    num_train_epochs            = EPOCHS,
    per_device_train_batch_size = BATCH_SIZE,
    per_device_eval_batch_size  = BATCH_SIZE * 2,  
    gradient_accumulation_steps = GRAD_ACCUM,      
    learning_rate               = LR,
    lr_scheduler_type           = "cosine",
    warmup_steps                = 200,
    weight_decay                = 0.01,
    eval_strategy               = "epoch",
    save_strategy               = "epoch",
    load_best_model_at_end      = True,
    metric_for_best_model       = "micro_f1",
    greater_is_better           = True,
    save_total_limit            = EPOCHS,  
    fp16                        = torch.cuda.is_available(),
    bf16                        = False,
    logging_steps               = 100,
    report_to                   = "none",
    seed                        = SEED,
    dataloader_num_workers      = 0,
    dataloader_pin_memory       = False,
)

best_model_saver = BestModelSaverCallback(
    output_dir   = OUTPUT_DIR,
    tokenizer    = tokenizer,
    label2id     = label2id,
    id2label     = id2label,
    max_len      = MAX_LEN,
    num_labels   = NUM_LABELS,
    e1_id        = e1_id,
    e2_id        = e2_id,
    hi_map_path  = HI_MAP_PATH,
    kn_map_path  = KN_MAP_PATH,
)

trainer = Trainer(
    model             = model,
    args              = training_args,
    train_dataset     = train_dataset,
    eval_dataset      = val_dataset,
    data_collator     = collator,
    compute_metrics   = compute_metrics,
    callbacks         = [
        EarlyStoppingCallback(early_stopping_patience=3),
        TqdmProgressCallback(),
        best_model_saver,       
    ],
)

trainer.train()

print("Saving final artefacts ...")

raw_model = trainer.model
if hasattr(raw_model, "module"):
    raw_model = raw_model.module

raw_model.encoder.save_pretrained(str(OUTPUT_DIR / "lora_adapter"))
tokenizer.save_pretrained(str(OUTPUT_DIR / "tokenizer"))
torch.save(raw_model.classifier.state_dict(),
           OUTPUT_DIR / "classifier_head.pt")
json.dump(
    {"max_length": MAX_LEN, "num_labels": NUM_LABELS,
     "e1_id": e1_id, "e2_id": e2_id},
    open(OUTPUT_DIR / "train_config.json", "w"), indent=2,
)
json.dump(label2id,
          open(OUTPUT_DIR / "label2id.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
json.dump({str(k): v for k, v in id2label.items()},
          open(OUTPUT_DIR / "id2label.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
for src, fname in [(HI_MAP_PATH, "hi_map.json"), (KN_MAP_PATH, "kn_map.json")]:
    if src.exists():
        shutil.copy(src, OUTPUT_DIR / fname)

print(f"\nAll artefacts saved → {OUTPUT_DIR}")
print(f"Files: {[p.name for p in sorted(OUTPUT_DIR.iterdir())]}")