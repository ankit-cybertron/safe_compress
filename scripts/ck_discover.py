
import sys, json, re, math, torch
import circuitkit as ck
algo, n, dtype, out, pairs, refuse, comply = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], sys.argv[5], sys.argv[6], sys.argv[7]
model = ck.load_model("CohereLabs/tiny-aya-global", dtype=dtype)
for name, p in model.named_parameters(): p.requires_grad_(name.endswith((".w", ".b")) and "ln" in name)
torch.cuda.empty_cache(); print("model loaded | GB allocated:", round(torch.cuda.memory_allocated() / 1e9, 2), "| trainable params (M):", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3), flush=True)
dp = {"language": "en", "pairs_path": pairs, "refuse": refuse, "comply": comply, "num_examples": n}
try:
    circuit = ck.discover(model, "multilingual_refusal", algorithm=algo, n_examples=n, batch_size=1, sparsity=0.3, scope="both", output_path=out + "/circuit.pt", chat_template_mode="off", data_params=dp)
except Exception:
    print("peak GB at failure:", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True); raise
vals = [float(v) for v in circuit.scores.values()]
if not vals or any(not math.isfinite(v) for v in vals) or max(abs(v) for v in vals) == 0:
    print("DEGENERATE SCORES: n =", len(vals), "| non-finite =", sum(not math.isfinite(v) for v in vals)); sys.exit(3)
layers = {}
for name, s in circuit.scores.items():
    m = re.match(r"[Aa](\d+)\.[hH]?(\d+)$", name); kind = "attn"
    if not m: m = re.match(r"MLP\s*(\d+)$", name, re.I); kind = "mlp"
    if not m: continue
    d = layers.setdefault(int(m.group(1)), {"attn": 0.0, "mlp": 0.0}); d[kind] += abs(float(s))
if not layers:
    print("NO LAYERS PARSED; example score keys:", list(circuit.scores)[:8]); sys.exit(2)
json.dump({"algo": algo, "dtype": dtype, "n_examples": n, "layers": layers, "n_nodes": len(circuit.scores), "peak_gb": torch.cuda.max_memory_allocated() / 1e9}, open(out + "/ck_layer_scores.json", "w"))
print("layers scored:", len(layers), "| nodes:", len(circuit.scores), "| peak GB:", round(torch.cuda.max_memory_allocated() / 1e9, 2))
