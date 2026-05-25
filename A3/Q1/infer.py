import copy
import json
import sys
from pathlib import Path

import torch
import torch.nn as nn
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


if len(sys.argv) < 4:
    print("Usage: python infer.py <lang> <test_file_path> <output_dir>")
    sys.exit(1)

LANG       = sys.argv[1].lower()
TEST_PATH  = Path(sys.argv[2]).expanduser().absolute()
OUTPUT_DIR = Path(sys.argv[3]).expanduser().absolute()

if LANG not in ("en", "hi", "kn"):
    print(f"WARNING: Task 1 supports en/hi/kn only — got '{LANG}'")

MODEL_NAME = "Qwen/Qwen2.5-1.5B"
BATCH_SIZE = 16   
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"\nTask 1 Inference | lang={LANG} | device={DEVICE}")
print(f"  Output dir : {OUTPUT_DIR}")


def load_jsonl(path):
    data = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:    data.append(json.loads(line))
                except: pass
    return data


def safe_load_map(path):
    if Path(path).exists():
        return json.load(open(path, encoding="utf-8"))
    return {}


def mark_entities(text, e1, e2):
    if e1 in text:
        text = text.replace(e1, f"[E1] {e1} [/E1]", 1)
    if e2 in text:
        text = text.replace(e2, f"[E2] {e2} [/E2]", 1)
    return text


def english_to_lang(label_en, lang, hi_map, kn_map):
    if lang == "en": return label_en
    if lang == "hi": return hi_map.get(label_en, label_en)
    if lang == "kn": return kn_map.get(label_en, label_en)
    return label_en



class REClassifier(nn.Module):

    def __init__(self, encoder, num_labels, e1_id, e2_id, pad_id):
        super().__init__()
        self.encoder = encoder
        self.e1_id   = e1_id
        self.e2_id   = e2_id
        self.pad_id  = pad_id
        hidden       = encoder.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_labels),
        )

    def _get_entity_rep(self, hidden_states, input_ids, token_id):
        reps = []
        for i in range(hidden_states.size(0)):
            pos = (input_ids[i] == token_id).nonzero(as_tuple=True)[0]
            if len(pos) > 0:
                reps.append(hidden_states[i, pos[0].item(), :])
            else:
                non_pad = (input_ids[i] != self.pad_id).nonzero(as_tuple=True)[0]
                reps.append(hidden_states[i, non_pad[-1].item(), :])
        return torch.stack(reps)

    def forward(self, input_ids, attention_mask):
        out    = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden = out.hidden_states[-1]
        e1_rep = self._get_entity_rep(hidden, input_ids, self.e1_id)
        e2_rep = self._get_entity_rep(hidden, input_ids, self.e2_id)
        pair   = torch.cat([e1_rep, e2_rep], dim=-1).float()
        return self.classifier(pair)



print("\nLoading artifacts ...")

train_config = json.load(open(OUTPUT_DIR / "train_config.json"))
MAX_LEN    = train_config["max_length"]
NUM_LABELS = train_config["num_labels"]
e1_id      = train_config["e1_id"]
e2_id      = train_config["e2_id"]
print(f"  max_length={MAX_LEN}  num_labels={NUM_LABELS}")
print(f"  [E1] id={e1_id}  [E2] id={e2_id}")

TOKENIZER_DIR = OUTPUT_DIR / "tokenizer"
tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

id2label = {
    int(k): v
    for k, v in json.load(
        open(OUTPUT_DIR / "id2label.json", encoding="utf-8")
    ).items()
}

hi_map = safe_load_map(OUTPUT_DIR / "hi_map.json")
kn_map = safe_load_map(OUTPUT_DIR / "kn_map.json")

print(f"Loading base model {MODEL_NAME} ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
)
base_model.config.output_hidden_states = True
base_model.resize_token_embeddings(len(tokenizer))

LORA_DIR   = OUTPUT_DIR / "lora_adapter"
print(f"Applying LoRA adapter from {LORA_DIR} ...")
lora_model = PeftModel.from_pretrained(base_model, str(LORA_DIR))

model = REClassifier(
    encoder    = lora_model,
    num_labels = NUM_LABELS,
    e1_id      = e1_id,
    e2_id      = e2_id,
    pad_id     = tokenizer.pad_token_id,
)
model.classifier.load_state_dict(
    torch.load(
        OUTPUT_DIR / "classifier_head.pt",
        map_location=DEVICE,
        weights_only=True,
    )
)
model.to(DEVICE)
model.eval()
print(f"  Model ready on {DEVICE}")


def tokenize_batch(texts):
    encodings = [
        tokenizer(t, truncation=True, max_length=MAX_LEN)
        for t in texts
    ]
    max_len = max(len(e["input_ids"]) for e in encodings)
    pad_id  = tokenizer.pad_token_id

    input_ids_out  = []
    attn_mask_out  = []
    for e in encodings:
        pad_len = max_len - len(e["input_ids"])
        input_ids_out.append(e["input_ids"]      + [pad_id] * pad_len)
        attn_mask_out.append(e["attention_mask"] + [0]      * pad_len)

    return (
        torch.tensor(input_ids_out, dtype=torch.long),
        torch.tensor(attn_mask_out, dtype=torch.long),
    )


print(f"\nLoading test file: {TEST_PATH} ...")
test_data = load_jsonl(TEST_PATH)
print(f"  {len(test_data)} entries")

flat_texts = []
flat_index = []  

for entry_idx, entry in enumerate(test_data):
    sent = entry["sentText"]
    for mention_idx, mention in enumerate(entry["relationMentions"]):
        em1  = mention["em1Text"]
        em2  = mention["em2Text"]
        text = f"[{LANG.upper()}] " + mark_entities(sent, em1, em2)
        flat_texts.append(text)
        flat_index.append((entry_idx, mention_idx))

print(f"  {len(flat_texts)} entity pairs to classify")



print("\nRunning inference ...")
all_pred_ids = []

with torch.no_grad():
    for start in tqdm(range(0, len(flat_texts), BATCH_SIZE), desc="Batches"):
        batch_texts          = flat_texts[start: start + BATCH_SIZE]
        input_ids, attn_mask = tokenize_batch(batch_texts)
        logits = model(input_ids.to(DEVICE), attn_mask.to(DEVICE))
        all_pred_ids.extend(logits.argmax(dim=-1).cpu().tolist())


out_entries = copy.deepcopy(test_data)

for (entry_idx, mention_idx), pred_id in zip(flat_index, all_pred_ids):
    label_en  = id2label[pred_id]
    label_out = english_to_lang(label_en, LANG, hi_map, kn_map)
    out_entries[entry_idx]["relationMentions"][mention_idx]["label"] = label_out

out_path = OUTPUT_DIR / f"Q1_{LANG}.jsonl"
with open(out_path, "w", encoding="utf-8") as f:
    for entry in out_entries:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

print(f"\nSaved → {out_path}  ({len(out_entries)} entries)")