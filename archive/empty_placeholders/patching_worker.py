
import sys, json, time, random, torch
from collections import defaultdict
data_path, outp, MAXP, MAXLEN, BS, MID, SEED = sys.argv[1:8]; MAXP, MAXLEN, BS, SEED = int(MAXP), int(MAXLEN), int(BS), int(SEED)
from transformers import AutoTokenizer, AutoModelForCausalLM
tok = AutoTokenizer.from_pretrained(MID, trust_remote_code=True)
m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, device_map={"": 0}, trust_remote_code=True).eval()
for p in m.parameters(): p.requires_grad_(False)
L = len(m.model.layers); BOS = tok.bos_token_id
REF = {"en": [15817, 55], "hi": [150617, 59007]}; COMP = 213          # first-token ids measured by the first-token probe: "I'm"/"I" (en), "mujhe"/"main" (hi) vs newline
MARK = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
def short_ids(p):                                                      # short format: user turn only (the default system preamble is dropped), BOS first
    t = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    return [BOS] + tok(t[t.rindex(MARK):], add_special_tokens=False).input_ids
rows = [json.loads(l) for l in open(data_path, encoding="utf-8") if l.strip()]
rng = random.Random(SEED + 7); pairs = []
for lang in ("en", "hi"):
    byl = {"harmful": defaultdict(list), "unharmful": defaultdict(list)}
    for r in sorted((x for x in rows if x["language"] == lang), key=lambda x: x["record_id"]):
        ids = short_ids(r["prompt"])
        if len(ids) <= MAXLEN: byl[r["label"]][len(ids)].append(ids)
    cand = []
    for n in sorted(set(byl["harmful"]) & set(byl["unharmful"])):
        h, b = byl["harmful"][n][:], byl["unharmful"][n][:]; rng.shuffle(h); rng.shuffle(b)
        cand += [(hh, bb) for hh, bb in zip(h, b)]
    rng.shuffle(cand)
    for i, c in enumerate(cand[:MAXP]): pairs.append({"lang": lang, "half": i % 2, "h": c[0], "b": c[1]})
print("pairs per language:", {l: sum(p["lang"] == l for p in pairs) for l in ("en", "hi")}, flush=True)
ST = {"cap": None, "patch": set(), "src": None}
def attn_hook(l):
    def f(mod, inp, out):
        tup = isinstance(out, tuple); o = out[0] if tup else out
        if ST["cap"] is not None: ST["cap"]["attn"][l] = o.detach()
        if ("attn", l) in ST["patch"]: return ((ST["src"]["attn"][l],) + tuple(out[1:])) if tup else ST["src"]["attn"][l]
    return f
def mlp_hook(l):
    def f(mod, inp, out):
        if ST["cap"] is not None: ST["cap"]["mlp"][l] = out.detach()
        if ("mlp", l) in ST["patch"]: return ST["src"]["mlp"][l]
    return f
for l, layer in enumerate(m.model.layers):
    layer.self_attn.register_forward_hook(attn_hook(l)); layer.mlp.register_forward_hook(mlp_hook(l))
def gap(ids, lang):
    with torch.inference_mode(): lg = m(input_ids=ids, logits_to_keep=1).logits[:, -1].float()
    return torch.logsumexp(lg[:, REF[lang]], -1) - lg[:, COMP]
def run(ids, lang, patch=(), src=None, cap=False):
    ST["cap"] = {"attn": {}, "mlp": {}} if cap else None; ST["patch"] = set(patch); ST["src"] = src
    g = gap(ids, lang); c = ST["cap"]; ST["cap"] = None; ST["patch"] = set(); ST["src"] = None
    return g, c
COMPS = {"attn": lambda l: [("attn", l)], "mlp": lambda l: [("mlp", l)], "block": lambda l: [("attn", l), ("mlp", l)]}
acc = defaultdict(lambda: [0.0, 0.0]); base = defaultdict(list); nvalid = defaultdict(int); t0 = time.time()
groups = defaultdict(list)
for p in pairs: groups[(p["lang"], p["half"], len(p["h"]))].append(p)
nb = 0
for (lang, half, n), ps in groups.items():
    for i in range(0, len(ps), BS):
        b = ps[i:i + BS]; H = torch.tensor([p["h"] for p in b], device="cuda:0"); Bn = torch.tensor([p["b"] for p in b], device="cuda:0")
        gh, ch = run(H, lang, cap=True); gb, cb = run(Bn, lang, cap=True)
        den = gh - gb; ok = den >= 2.0                                  # keep pairs where harmful and unharmful behave differently
        base[(lang, "harmful")] += gh.tolist(); base[(lang, "unharmful")] += gb.tolist(); nvalid[(lang, half)] += int(ok.sum())
        if not ok.any(): continue
        for l in range(L):
            for cname, fn in COMPS.items():
                pt = fn(l)
                gp, _ = run(H, lang, patch=pt, src=cb)                    # denoising: unharmful activation into the harmful run
                gn, _ = run(Bn, lang, patch=pt, src=ch)                   # noising: harmful activation into the unharmful run
                k1, k2 = (lang, half, "denoise", cname, l), (lang, half, "noise", cname, l)
                acc[k1][0] += float(((gh - gp) * ok).sum()); acc[k1][1] += float((den * ok).sum())
                acc[k2][0] += float(((gn - gb) * ok).sum()); acc[k2][1] += float((den * ok).sum())
        nb += 1
        if nb % 5 == 0: print(f"batches {nb} | {round(time.time() - t0)} s", flush=True)
res = {"layers": L, "n_pairs": {f"{k[0]}/half{k[1]}": v for k, v in nvalid.items()}, "baseline_gap_mean": {f"{k[0]}/{k[1]}": sum(v) / len(v) for k, v in base.items()},
       "scores": {"|".join(map(str, k)): (v[0] / v[1] if v[1] else None) for k, v in acc.items()}, "seconds": round(time.time() - t0), "refusal_tokens": REF, "compliance_token": COMP, "max_len": MAXLEN, "seed": SEED, "model": MID}
json.dump(res, open(outp, "w"), indent=1); print("done", res["seconds"], "s")
