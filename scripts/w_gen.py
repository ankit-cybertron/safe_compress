
import sys, json, time, gc
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM
from lab_common import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2]); ARMS = {a[0]: a for a in json.load(open(ROOT / "scores" / "arms.json"))["arms"]}; GEN = json.load(open(ROOT / "scores" / "gen_arms.json")); CFG = json.load(open(ROOT / "scores" / "eval_cfg.json"))
todo = [ARMS[n] for i, n in enumerate(GEN) if n in ARMS and i % nsh == shard and not (ROOT / "responses" / f"{n}.json").exists()]; print(f"shard {shard}: {[a[0] for a in todo]}", flush=True)
if not todo: sys.exit(0)
STOP = [3, 6, 261001]; chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True); enc = lambda p: tok(chat(p), add_special_tokens=False).input_ids
def pick(rows, n, salt):
    rows = sorted(rows, key=lambda r: r["id"]); random.Random(f"{SEED}-{salt}").shuffle(rows); out = []
    for r in rows:
        ids = enc(r["prompt"])
        if len(ids) <= 512: out.append({"id": r["id"], "dataset": r["dataset"], "lang": r["lang"], "ids": ids})
        if len(out) == n: break
    return out
N = CFG["n_group"]; hb = read("harmbench_standard"); sr = read("strongreject"); xs = [r for r in read("xstest") if r["label"] == "benign"]; ind = read("indicsafe_hindi")
SAFE = [dict(x, group="harmful_en") for x in pick(hb, N // 2, "hb") + pick(sr, N - N // 2, "sr")] + [dict(x, group="harmful_hi") for x in pick([r for r in ind if r["label"] == "harmful"], N, "ih")] \
     + [dict(x, group="benign_en") for x in pick(xs, N, "xs")] + [dict(x, group="benign_hi") for x in pick([r for r in ind if r["label"] == "benign"], N, "ib")]
def quant(layer, bits):
    for name, mod in list(layer.named_modules()):
        if not isinstance(mod, torch.nn.Linear): continue
        w = mod.weight.data.to(torch.float16)
        if bits == 4:
            new = bnb.nn.Linear4bit(mod.in_features, mod.out_features, bias=(mod.bias is not None), quant_type="nf4", compute_dtype=torch.float16); new.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
        else:
            new = bnb.nn.Linear8bitLt(mod.in_features, mod.out_features, bias=(mod.bias is not None), has_fp16_weights=False, threshold=6.0); new.weight = bnb.nn.Int8Params(w, requires_grad=False, has_fp16_weights=False)
        if mod.bias is not None: new.bias = torch.nn.Parameter(mod.bias.data.clone().to(torch.float16))
        pn, _, at = name.rpartition("."); par = layer
        if pn:
            for part in pn.split("."): par = getattr(par, part)
        setattr(par, at, new)
def build(prot, pbits, rbits):
    if pbits == 16 and rbits == 16: return AutoModelForCausalLM.from_pretrained(MID, token=HF, dtype=torch.float16, device_map={"": 0}).eval()
    m = AutoModelForCausalLM.from_pretrained(MID, token=HF, dtype=torch.float16, low_cpu_mem_usage=True, device_map=None).eval()
    for i, Ly in enumerate(m.model.layers):
        b = pbits if i in prot else rbits
        if b < 16: quant(Ly, b)
    m.model.embed_tokens = m.model.embed_tokens.to(dev)
    for Ly in m.model.layers: Ly.to(dev)
    m.model.norm = m.model.norm.to(dev); m.lm_head = m.lm_head.to(dev); return m
for arm, prot, pbits, rbits in todo:
    t0 = time.time(); m = build(set(prot), pbits, rbits); m.config.use_cache = True; seqs = [x["ids"] for x in SAFE]; rows = [None] * len(SAFE)
    for idx in batches(seqs, max_bs=16, budget=9000):
        ids, am, _ = pad_left([seqs[i] for i in idx]); n0 = ids.shape[1]
        with torch.inference_mode(): out = m.generate(input_ids=ids, attention_mask=am, max_new_tokens=48, do_sample=False, eos_token_id=STOP, pad_token_id=0)
        for r, i in enumerate(idx):
            g = out[r, n0:].tolist(); k = next((j for j, x in enumerate(g) if x in STOP), None); it = SAFE[i]
            rows[i] = {"id": it["id"], "group": it["group"], "lang": it["lang"], "dataset": it["dataset"], "text": tok.decode(g if k is None else g[:k], skip_special_tokens=True)}
    json.dump(rows, open(ROOT / "responses" / f"{arm}.json", "w"), ensure_ascii=False); print(arm, "DONE", round(time.time() - t0), "s", flush=True); del m; gc.collect(); torch.cuda.empty_cache()
