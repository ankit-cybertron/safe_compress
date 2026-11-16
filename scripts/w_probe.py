
import sys, json
from lab_common import *
m = load_fp16(); L = len(m.model.layers); cfg = m.config; lt = getattr(cfg, "layer_types", None)
ANAT = pd.DataFrame([{"layer": i, "type": (lt[i] if lt else ("full_attention" if i % 4 == 3 else "sliding_attention")), "params_M": sum(p.numel() for p in ly.parameters()) / 1e6,
                      "attn_M": sum(p.numel() for p in ly.self_attn.parameters()) / 1e6, "mlp_M": sum(p.numel() for p in ly.mlp.parameters()) / 1e6} for i, ly in enumerate(m.model.layers)])
ANAT.to_csv(ROOT / "scores" / "anatomy.csv", index=False)
print(f"layers {L} | hidden {cfg.hidden_size} | heads {cfg.num_attention_heads} | kv heads {getattr(cfg, 'num_key_value_heads', '?')} | vocab {cfg.vocab_size} | tied embeddings {getattr(cfg, 'tie_word_embeddings', '?')}")
print(ANAT.type.value_counts().to_dict(), "| params/layer (M):", round(ANAT.params_M.iloc[0], 1), "| embed (M):", round(m.model.embed_tokens.weight.numel() / 1e6, 1), "| full-attention layers:", ANAT.layer[ANAT.type.str.startswith("full")].tolist())
print({n: tuple(mod.weight.shape) for n, mod in m.model.layers[0].named_modules() if isinstance(mod, torch.nn.Linear)})
DISC = read("polyguardmix_clean"); rng = random.Random(SEED + 1); FT = {}
for lg in ("en", "hi"):
    for lab in ("harmful", "unharmful"):
        rows = [r for r in DISC if r["lang"] == lg and r["label"] == lab]; rng.shuffle(rows); seqs = [short_ids(r["prompt"]) for r in rows[:120]]; c = Counter()
        for idx, lgt in forward_last(m, seqs): c.update(lgt.argmax(-1).tolist())
        FT[(lg, lab)] = c
REF, COMP = {}, {}
for lg in ("en", "hi"):
    h, b = FT[(lg, "harmful")], FT[(lg, "unharmful")]; nh, nb = sum(h.values()), sum(b.values())
    sc = {t: (h[t] / nh + 1e-3) / (b.get(t, 0) / nb + 1e-3) for t in h if h[t] >= 3}
    REF[lg] = sorted(sc, key=sc.get, reverse=True)[:3]; COMP[lg] = [t for t, _ in b.most_common(8) if t not in REF[lg]][:3]
    print(f"[{lg}] refusal first-tokens   (id, text, count on harmful, count on benign):", [(t, tok.decode([t]), h[t], b.get(t, 0)) for t in REF[lg]])
    print(f"[{lg}] compliance first-tokens (id, text, count on benign, count on harmful):", [(t, tok.decode([t]), b[t], h.get(t, 0)) for t in COMP[lg]])
json.dump({"REF": REF, "COMP": COMP}, open(ROOT / "scores" / "first_tokens.json", "w"))
