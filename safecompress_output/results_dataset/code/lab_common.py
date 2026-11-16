
import os, sys, json, re, time, random, gc
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np, pandas as pd, torch
ROOT = Path("/kaggle/working/layerlab"); SCRATCH = Path("/kaggle/temp/layerlab") if Path("/kaggle/temp").exists() else Path("/tmp/layerlab"); DATA = SCRATCH / "data"
MID = "CohereLabs/tiny-aya-global"; SEED = 20261101; HF = os.environ.get("HF_TOKEN") or None; dev = "cuda:0"
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MID, token=HF); BOS = tok.bos_token_id; MARK = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
def load_fp16():
    mm = AutoModelForCausalLM.from_pretrained(MID, token=HF, dtype=torch.float16, device_map={"": 0}).eval()
    for p in mm.parameters(): p.requires_grad_(False)
    return mm
def read(key): return [json.loads(l) for l in open(DATA / f"{key}.jsonl", encoding="utf-8") if l.strip()]
def short_ids(p):
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    return [BOS] + tok(t[t.rindex(MARK):], add_special_tokens=False).input_ids
def pad_left(seqs):
    n = max(len(s) for s in seqs); ids = torch.zeros((len(seqs), n), dtype=torch.long); am = torch.zeros((len(seqs), n), dtype=torch.long)
    for r, s in enumerate(seqs): ids[r, n - len(s):] = torch.tensor(s); am[r, n - len(s):] = 1
    ids, am = ids.to(dev), am.to(dev)
    return ids, am, (am.cumsum(-1) - 1).clamp(min=0)
def batches(seqs, max_bs=16, budget=5000):
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i])); s = 0
    while s < len(order):
        b = max(1, min(max_bs, budget // len(seqs[order[min(s + max_bs, len(order)) - 1]]))); yield order[s:s + b]; s += b
def forward_last(model, seqs, max_bs=16):
    for idx in batches(seqs, max_bs):
        ids, am, pos = pad_left([seqs[i] for i in idx])
        with torch.inference_mode(): lg = model(input_ids=ids, attention_mask=am, position_ids=pos, logits_to_keep=1).logits[:, -1].float()
        yield idx, lg
def load_ref():
    d = json.load(open(ROOT / "scores" / "first_tokens.json"))
    return {k: torch.tensor(v, device=dev) for k, v in d["REF"].items()}, {k: torch.tensor(v, device=dev) for k, v in d["COMP"].items()}
