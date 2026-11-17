# /// script
# requires-python = ">=3.11"
# dependencies = ["circuitkit", "huggingface_hub", "numpy"]
# ///
import os, sys, json, re, math, csv
import torch, circuitkit as ck
from huggingface_hub import HfApi, hf_hub_download
from transformers import AutoTokenizer
HF = os.environ["HF_TOKEN"]; STORE = sys.argv[1]; MID = "CohereLabs/tiny-aya-global"
tk = AutoTokenizer.from_pretrained(MID, token=HF)
groups = json.load(open(hf_hub_download(STORE, "scores/patch_groups.json", repo_type="dataset", token=HF))); ft = json.load(open(hf_hub_download(STORE, "scores/first_tokens.json", repo_type="dataset", token=HF)))
def opener(tid, default):
    s = tk.decode([tid]).strip(); return s if s and tk.encode(s, add_special_tokens=False)[:1] == [tid] else default
REFUSE, COMPLY = opener(ft["REF"]["en"][0], "I"), opener(ft["COMP"]["en"][0], "Sure")
if REFUSE == COMPLY: REFUSE, COMPLY = "I", "Sure"
MARK = "<|START_OF_TURN_TOKEN|><|USER_TOKEN|>"; pairs = []
for g in groups:
    if g["lang"] != "en": continue
    for p in g["pairs"]:
        h, b = tk.decode(p["h"][1:]), tk.decode(p["b"][1:])
        if h.startswith(MARK) and b.startswith(MARK) and len(tk(h, add_special_tokens=False).input_ids) == len(tk(b, add_special_tokens=False).input_ids) and h != b: pairs.append((h, b))
with open("pairs.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f); w.writerow(["harmful", "harmless"]); w.writerows(pairs[:200])
print(len(pairs), "pairs | openers", REFUSE, COMPLY, flush=True)
def run(algo, dtype, train):
    model = ck.load_model(MID, dtype=dtype)
    if train == "ln":
        for name, p in model.named_parameters(): p.requires_grad_(name.endswith((".w", ".b")) and "ln" in name)
    print("trainable params (M):", round(sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6, 3), flush=True)
    dp = {"language": "en", "pairs_path": "pairs.csv", "refuse": REFUSE, "comply": COMPLY, "num_examples": 64}
    c = ck.discover(model, "multilingual_refusal", algorithm=algo, n_examples=64, batch_size=2, sparsity=0.3, scope="both", output_path="circuit.pt", chat_template_mode="off", data_params=dp)
    return c
circuit = None
for algo, dtype, train in (("eap-ig", "float16", "all"), ("eap-ig", "float16", "ln"), ("eap-ig", "bfloat16", "ln"), ("eap", "float16", "ln")):
    try:
        circuit = run(algo, dtype, train); print("SUCCESS:", algo, dtype, train, "| peak GB", round(torch.cuda.max_memory_allocated() / 1e9, 2), flush=True); break
    except Exception as e:
        print("attempt failed:", algo, dtype, train, repr(e)[:300], flush=True); torch.cuda.empty_cache()
assert circuit is not None, "all attempts failed"
vals = [float(v) for v in circuit.scores.values()]; assert vals and all(math.isfinite(v) for v in vals) and max(abs(v) for v in vals) > 0, "degenerate scores"
layers = {}
for name, s in circuit.scores.items():
    m = re.match(r"[Aa](\d+)\.[hH]?(\d+)$", name); kind = "attn"
    if not m: m = re.match(r"MLP\s*(\d+)$", name, re.I); kind = "mlp"
    if not m: continue
    d = layers.setdefault(int(m.group(1)), {"attn": 0.0, "mlp": 0.0}); d[kind] += abs(float(s))
assert layers, "no layers parsed; example keys: " + str(list(circuit.scores)[:8])
json.dump({"algo": algo, "dtype": dtype, "layers": layers, "n_nodes": len(circuit.scores), "peak_gb": torch.cuda.max_memory_allocated() / 1e9}, open("ck_layer_scores.json", "w"))
api = HfApi(token=HF); api.upload_file(path_or_fileobj="ck_layer_scores.json", path_in_repo="scores/ck_layer_scores.json", repo_id=STORE, repo_type="dataset", commit_message="CircuitKIT layer scores (HF Job)")
for p in os.listdir("."):
    if p.endswith("_scores.json") and p != "ck_layer_scores.json": api.upload_file(path_or_fileobj=p, path_in_repo="scores/ck_scores_sidecar.json", repo_id=STORE, repo_type="dataset"); break
print("uploaded scores/ck_layer_scores.json", flush=True)
