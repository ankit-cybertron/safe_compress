
import sys, json, re, gc, math, torch
import circuitkit as ck
cfg = json.loads(sys.argv[1]); out = sys.argv[2]
print("visible gpus:", torch.cuda.device_count(), "| config:", cfg, flush=True)
model = ck.load_model("CohereLabs/tiny-aya-global", dtype=cfg.get("dtype", "float16"), algorithm=cfg["algo"], **({"device": cfg["device"]} if cfg.get("device") else {}))
tk = model.tokenizer; MARK = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"
try:                                                            # Tiny Aya's default chat template prepends a long system preamble to every prompt; the task would carry it into every position of the attribution buffers
    probe = tk.apply_chat_template([{"role": "user", "content": "XQXQ"}], tokenize=False, add_generation_prompt=True)
    pre, post = probe[probe.rindex(MARK):].split("XQXQ")
    n_before = len(tk(probe, add_special_tokens=False).input_ids)
    tk.chat_template = "{{ bos_token }}" + pre + "{{ messages[-1]['content'] }}" + post
    probe2 = tk.apply_chat_template([{"role": "user", "content": "XQXQ"}], tokenize=False, add_generation_prompt=True)
    print("chat template shortened: overhead tokens", n_before, "->", len(tk(probe2, add_special_tokens=False).input_ids), "| new text:", repr(probe2), flush=True)
except Exception as e: print("template shortening skipped:", repr(e)[:200], flush=True)
for p in model.parameters(): p.requires_grad_(False)          # no parameter gradients (saves ~6.7 GB of grad buffers)
def _grad_on(t, hook):                                         # but keep the activation graph alive: embedding output is the graph root
    return t.detach().requires_grad_(True) if (torch.is_floating_point(t) and not t.requires_grad) else t
model.hook_embed.add_hook(_grad_on, dir="fwd", is_permanent=True)
try:                                                            # TransformerLens keeps W_E and W_U as two copies; Tiny Aya ties them. If identical, share one copy (~1 GB back).
    WE, WU = model.embed.W_E, model.unembed.W_U
    if WE.shape == WU.T.shape and torch.equal(WE.data, WU.data.T):
        model.unembed.W_U = torch.nn.Parameter(WE.data.T, requires_grad=False); print("tied W_U to W_E (identical weights), freed", round(WE.numel() * 2 / 1e9, 2), "GB", flush=True)
    else: print("W_U differs from W_E.T, left untouched", flush=True)
except Exception as e: print("tie skipped:", repr(e)[:200], flush=True)
gc.collect(); torch.cuda.empty_cache()
try: print("GPU memory after load: allocated GB", round(torch.cuda.memory_allocated() / 1e9, 2), flush=True)
except Exception: pass
try: torch.cuda.memory._record_memory_history(max_entries=200000)
except Exception as e: print("memory history unavailable:", repr(e)[:100], flush=True)
def _who_holds():                                               # on failure: biggest live allocations grouped by the code that made them
    try:
        agg = {}
        for seg in torch.cuda.memory._snapshot()["segments"]:
            for b in seg["blocks"]:
                if b["state"] != "active_allocated": continue
                fr = [f for f in b.get("frames", []) if "site-packages/torch/" not in f["filename"]][:3] or b.get("frames", [])[:2]
                k = " <- ".join(f"{f['filename'].split('site-packages/')[-1]}:{f['line']}" for f in fr) or "unknown"
                agg[k] = agg.get(k, 0) + b["size"]
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1])[:8]: print("TOPALLOC", round(v / 1e9, 2), "GB", k, flush=True)
    except Exception as e: print("TOPALLOC snapshot failed:", repr(e)[:150], flush=True)
try:
  circuit = ck.discover(model, "multilingual_refusal", algorithm=cfg["algo"], n_examples=cfg["n"], batch_size=1, sparsity=0.3, scope=cfg["scope"], output_path=out + "/circuit.pt", **cfg.get("extra", {}))
except Exception:
  _who_holds(); raise
names = list(circuit.scores.keys()); print("sample node names:", names[:8], "| n nodes:", len(names), flush=True)
layers = {}; bad = 0
for name, s in circuit.scores.items():
    low = str(name).lower(); nums = re.findall(r"\d+", str(name)); v = float(s)
    if not nums or not math.isfinite(v): bad += (not math.isfinite(v)); continue
    kind = "mlp" if ("mlp" in low or re.match(r"m[\s_.]?\d", low)) else ("attn" if (low.startswith("a") or "attn" in low or "head" in low) else None)
    if kind is None or int(nums[0]) > 200: continue
    d = layers.setdefault(int(nums[0]), {"attn": 0.0, "mlp": 0.0}); d[kind] += abs(v)
assert layers and sum(d["attn"] + d["mlp"] for d in layers.values()) > 0, f"no usable layer scores (non-finite scores: {bad}); see 'sample node names' above"
json.dump({"algo": cfg["algo"], "scope": cfg["scope"], "n_examples": cfg["n"], "layers": layers, "n_nodes": len(names), "nonfinite": bad}, open(out + "/ck_layer_scores.json", "w"))
print("layers scored:", len(layers), "| nodes:", len(names), "| non-finite:", bad)
