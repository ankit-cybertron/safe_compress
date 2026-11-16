
import sys, json, time
import bitsandbytes.functional as BF
from lab_common import *
shard, nsh = int(sys.argv[1]), int(sys.argv[2])
m = load_fp16(); L = len(m.model.layers); REF_T, COMP_T = load_ref(); mylayers = list(range(L))[shard::nsh]
DISC = read("polyguardmix_clean"); rng = random.Random(SEED + 3); N_SAFE, N_MM = 48, 96; cal_safe = []
for lg in ("en", "hi"):
    for lab in ("harmful", "unharmful"):
        rows = [r for r in DISC if r["lang"] == lg and r["label"] == lab]; rng.shuffle(rows); cal_safe += [(short_ids(r["prompt"]), lg, lab) for r in rows[:N_SAFE]]
LAB = {"A": 375, "B": 437, "C": 395, "D": 441}
def mmlu_seq(r): return short_ids(r["question"] + "\n" + "\n".join(f"{l}. {o}" for l, o in zip("ABCD", r["options"])) + "\n\nAnswer with only the single letter A, B, C, or D.\nAnswer:")
cal_mm = [(mmlu_seq(r), r["lang"], r["gold"]) for lg in ("en", "hi") for r in read(f"global_mmlu_{lg}")[500:500 + N_MM]]
SEQ = [c[0] for c in cal_safe] + [c[0] for c in cal_mm]; META = cal_safe + cal_mm; NS = len(cal_safe); LANGS = [c[1] for c in META]; LETTER_IDX = torch.tensor([LAB[k] for k in "ABCD"], device=dev)
if shard == 0: print("letter ids decode:", {k: tok.decode([v]) for k, v in LAB.items()}, flush=True)
def gap_row(lg_row, lang): return float(torch.logsumexp(lg_row[REF_T[lang]], -1) - torch.logsumexp(lg_row[COMP_T[lang]], -1))
def evaluate(model, base_logp=None):
    store = torch.empty((len(SEQ), model.config.vocab_size), dtype=torch.float32) if base_logp is None else None
    kl = np.zeros(len(SEQ)); gap = np.zeros(NS); pred = [""] * len(SEQ)
    for idx, lg in forward_last(model, SEQ):
        lp = torch.log_softmax(lg, -1)
        if store is not None: store[idx] = lp.cpu()
        else: b = base_logp[idx].to(dev); kl[idx] = (b.exp() * (b - lp)).sum(-1).cpu().numpy()
        pl = lg[:, LETTER_IDX].argmax(-1).tolist()
        for j, i in enumerate(idx):
            pred[i] = "ABCD"[pl[j]]
            if i < NS: gap[i] = gap_row(lg[j], LANGS[i])
    return store, kl, gap, pred
base_logp, _, gap0, pred0 = evaluate(m); acc0 = np.mean([pred0[NS + j] == META[NS + j][2] for j in range(len(cal_mm))])
print(f"shard {shard}: layers {mylayers} | calibration {NS} safety + {len(cal_mm)} MMLU | FP16 calib MMLU acc {acc0:.3f}", flush=True)
def fake_nf4(w): q, st = BF.quantize_4bit(w.contiguous(), blocksize=64, quant_type="nf4"); return BF.dequantize_4bit(q, st).to(w.dtype)
def fake_int8(w): s = (w.abs().amax(1, keepdim=True).float() / 127).clamp(min=1e-8); return ((w.float() / s).round().clamp(-127, 127) * s).to(w.dtype)
SCHEMES = {"nf4_all": (fake_nf4, ("self_attn", "mlp")), "nf4_attn": (fake_nf4, ("self_attn",)), "nf4_mlp": (fake_nf4, ("mlp",)), "int8_all": (fake_int8, ("self_attn", "mlp"))}
hmask = np.array([META[i][2] == "harmful" for i in range(NS)]); bmask = np.array([META[i][2] == "unharmful" for i in range(NS)]); rows = []; t0 = time.time()
for l in mylayers:
    layer = m.model.layers[l]; lin = {n: mod for n, mod in layer.named_modules() if isinstance(mod, torch.nn.Linear)}; orig = {n: mod.weight.data.clone() for n, mod in lin.items()}
    errs, kurt, outl = [], [], []
    for n, w in orig.items():
        wf = w.float(); q = fake_nf4(w).float(); errs.append(float((wf - q).norm() / wf.norm())); z = (wf - wf.mean()) / wf.std(); kurt.append(float((z ** 4).mean() - 3)); outl.append(float(wf.abs().max() / wf.abs().median()))
    r = {"layer": l, "nf4_relerr": float(np.mean(errs)), "kurtosis": float(np.mean(kurt)), "outlier_ratio": float(np.mean(outl))}
    for sname, (fq, scopes) in SCHEMES.items():
        for n, mod in lin.items():
            if n.startswith(scopes): mod.weight.data.copy_(fq(orig[n]))
        _, kl, gap, pred = evaluate(m, base_logp)
        for n, mod in lin.items(): mod.weight.data.copy_(orig[n])
        r[f"kl_safety_{sname}"] = kl[:NS].mean(); r[f"kl_mmlu_{sname}"] = kl[NS:].mean(); r[f"dgap_harm_{sname}"] = (gap - gap0)[hmask].mean(); r[f"dgap_benign_{sname}"] = (gap - gap0)[bmask].mean()
        r[f"mmlu_acc_delta_{sname}"] = np.mean([pred[NS + j] == META[NS + j][2] for j in range(len(cal_mm))]) - acc0
    rows.append(r); print(f"layer {l} done | KL safety nf4 {r['kl_safety_nf4_all']:.4f} | dgap harm {r['dgap_harm_nf4_all']:+.2f} | {time.time() - t0:.0f}s", flush=True)
pd.DataFrame(rows).to_csv(ROOT / "scores" / f"sens_part{shard}.csv", index=False)
