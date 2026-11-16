
import sys, json, time
from lab_common import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2]); groups = json.load(open(ROOT / "scores" / "patch_groups.json"))[shard::nsh]
m = load_fp16(); L = len(m.model.layers); REF_T, COMP_T = load_ref(); BS, SC = 4, 256.0
gap = lambda lg, lang: torch.logsumexp(lg[:, REF_T[lang]], -1) - torch.logsumexp(lg[:, COMP_T[lang]], -1)
CAP = {"on": False, "attn": {}, "mlp": {}}
m.model.embed_tokens.register_forward_hook(lambda mod, inp, out: out.detach().requires_grad_(True) if CAP["on"] else out)   # params are frozen, so make the embedding output the root of the graph
def hook(kind, l):
    def f(mod, inp, out):
        if not CAP["on"]: return
        o = out[0] if isinstance(out, tuple) else out
        if o.requires_grad: o.retain_grad()
        CAP[kind][l] = o
    return f
for l, layer in enumerate(m.model.layers): layer.self_attn.register_forward_hook(hook("attn", l)); layer.mlp.register_forward_hook(hook("mlp", l))
def run(ids, lang):
    CAP.update(on=True, attn={}, mlp={})
    with torch.enable_grad():
        lg = m(input_ids=ids, logits_to_keep=1).logits[:, -1].float(); g = gap(lg, lang); (g.sum() * SC).backward()
    out = {k: {l: (v.detach(), (v.grad / SC) if v.grad is not None else None) for l, v in CAP[k].items()} for k in ("attn", "mlp")}
    CAP.update(on=False, attn={}, mlp={}); m.zero_grad(set_to_none=True); return g.detach(), out
acc = defaultdict(lambda: [0.0, 0.0]); den_sum = defaultdict(float); nvalid = defaultdict(int); nonfinite = 0; t0 = time.time(); nb = 0; tot = sum((len(g["pairs"]) + BS - 1) // BS for g in groups)
for g in groups:
    lang, ps = g["lang"], g["pairs"]
    for i in range(0, len(ps), BS):
        b = ps[i:i + BS]; H = torch.tensor([p["h"] for p in b], device=dev); Bn = torch.tensor([p["b"] for p in b], device=dev)
        gh, oh = run(H, lang); gb, ob = run(Bn, lang); den = gh - gb; ok = (den >= 2.0) & torch.isfinite(den); nb += 1
        if ok.any():
            den_sum[lang] += float(den[ok].sum()); nvalid[lang] += int(ok.sum())
            for kind in ("attn", "mlp"):
                for l in range(L):
                    ah, grh = oh[kind][l]; ab, grb = ob[kind][l]
                    if grh is None or grb is None: nonfinite += 1; continue
                    d_est = ((ab - ah).float() * grh.float()).sum((1, 2))     # predicted change of the harmful-run gap if this output is replaced by the benign one
                    n_est = ((ah - ab).float() * grb.float()).sum((1, 2))     # predicted change of the benign-run gap if replaced by the harmful one
                    bad = ~(torch.isfinite(d_est) & torch.isfinite(n_est)); nonfinite += int((bad & ok).sum()); keep = ok & ~bad
                    k = f"{lang}|{kind}|{l}"; acc[k][0] += float((-d_est * keep).sum()); acc[k][1] += float((n_est * keep).sum())   # denoise score = -d_est/den, noise score = n_est/den (same convention as Cell 5)
        print(f"batch {nb}/{tot} | {time.time() - t0:.0f}s", flush=True)
json.dump({"acc": dict(acc), "den": dict(den_sum), "nvalid": dict(nvalid), "nonfinite": nonfinite}, open(ROOT / "scores" / f"attr_part{shard}.json", "w"))
