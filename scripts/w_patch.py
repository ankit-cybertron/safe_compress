
import sys, json, time
from lab_common import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2]); groups = json.load(open(ROOT / "scores" / "patch_groups.json"))[shard::nsh]
m = load_fp16(); L = len(m.model.layers); REF_T, COMP_T = load_ref(); BS = 8
gap = lambda lg, lang: torch.logsumexp(lg[:, REF_T[lang]], -1) - torch.logsumexp(lg[:, COMP_T[lang]], -1)
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
for l, layer in enumerate(m.model.layers): layer.self_attn.register_forward_hook(attn_hook(l)); layer.mlp.register_forward_hook(mlp_hook(l))
def run(ids, lang, patch=(), src=None, cap=False):
    ST["cap"] = {"attn": {}, "mlp": {}} if cap else None; ST["patch"] = set(patch); ST["src"] = src
    with torch.inference_mode(): lg = m(input_ids=ids, logits_to_keep=1).logits[:, -1].float()
    g = gap(lg, lang); c = ST["cap"]; ST["cap"] = None; ST["patch"] = set(); ST["src"] = None; return g, c
COMPS = {"attn": lambda l: [("attn", l)], "mlp": lambda l: [("mlp", l)], "block": lambda l: [("attn", l), ("mlp", l)]}
acc = defaultdict(lambda: [0.0, 0.0]); nvalid = defaultdict(int); base = defaultdict(list); t0 = time.time(); nb_ = 0; tot = sum((len(g["pairs"]) + BS - 1) // BS for g in groups)
for g in groups:
    lang, half, ps = g["lang"], g["half"], g["pairs"]
    for i in range(0, len(ps), BS):
        b = ps[i:i + BS]; H = torch.tensor([p["h"] for p in b], device=dev); Bn = torch.tensor([p["b"] for p in b], device=dev)
        gh, ch = run(H, lang, cap=True); gb, cb = run(Bn, lang, cap=True); den = gh - gb; ok = den >= 2.0
        base[lang + "/harmful"] += gh.tolist(); base[lang + "/unharmful"] += gb.tolist(); nvalid[f"{lang}/half{half}"] += int(ok.sum())
        if ok.any():
            for l in range(L):
                for cname, fn in COMPS.items():
                    gp, _ = run(H, lang, patch=fn(l), src=cb); gn, _ = run(Bn, lang, patch=fn(l), src=ch)
                    k1, k2 = f"{lang}|{half}|denoise|{cname}|{l}", f"{lang}|{half}|noise|{cname}|{l}"
                    acc[k1][0] += float(((gh - gp) * ok).sum()); acc[k1][1] += float((den * ok).sum()); acc[k2][0] += float(((gn - gb) * ok).sum()); acc[k2][1] += float((den * ok).sum())
        nb_ += 1; print(f"batch {nb_}/{tot} | {time.time() - t0:.0f}s", flush=True)
json.dump({"acc": dict(acc), "nvalid": dict(nvalid), "base": dict(base)}, open(ROOT / "scores" / f"patch_part{shard}.json", "w"))
