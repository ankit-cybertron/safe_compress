"""Rebuild any SafeCompress arm from the upstream FP16 model + the frozen manifest (layer_selection.json).
Needs: a GPU (bitsandbytes quantizes when weights move to CUDA), access to CohereLabs/tiny-aya-global (gated; set HF_TOKEN in the environment, never in code).
Usage: python load_safecompress.py --arm E3 --manifest results/manifests/layer_selection.json
Arms: E0 FP16 | E1 uniform W4 | E2 uniform W8 | E3 circuit FP16+W4 | E4 least-effect FP16+W4 | E8 circuit W8+W4 | E9 least-effect W8+W4 | E_rand_FP16_W4_s0..9 | E_rand_W8_W4_s0..9"""
import argparse, json, hashlib, torch, bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer
MID = "CohereLabs/tiny-aya-global"; BITS = {"FP16": 16, "W8": 8, "W4": 4}
def quantize_layer_inplace(layer, bits):
    for name, mod in list(layer.named_modules()):
        if not isinstance(mod, torch.nn.Linear): continue
        w = mod.weight.data.to(torch.float16)
        if bits == 4:
            new = bnb.nn.Linear4bit(mod.in_features, mod.out_features, bias=(mod.bias is not None), quant_type="nf4", compute_dtype=torch.float16)
            new.weight = bnb.nn.Params4bit(w, requires_grad=False, quant_type="nf4")
        else:
            new = bnb.nn.Linear8bitLt(mod.in_features, mod.out_features, bias=(mod.bias is not None), has_fp16_weights=False, threshold=6.0)
            new.weight = bnb.nn.Int8Params(w, requires_grad=False, has_fp16_weights=False)
        if mod.bias is not None: new.bias = torch.nn.Parameter(mod.bias.data.clone().to(torch.float16))
        parent_name, _, attr = name.rpartition("."); parent = layer
        if parent_name:
            for part in parent_name.split("."): parent = getattr(parent, part)
        setattr(parent, attr, new)
def build_arm(arm, manifest, device="cuda:0"):
    spec = manifest["unified_matrix"][arm]; prot, pdt, bdt = spec["protected_layers"], spec["protect_dtype"], spec["base_dtype"]
    m = AutoModelForCausalLM.from_pretrained(MID, dtype=torch.float16, low_cpu_mem_usage=True, device_map=None).eval()
    for i, layer in enumerate(m.model.layers):
        bits = 16 if prot == "ALL" else (BITS[pdt] if i in prot else BITS[bdt])
        if bits < 16: quantize_layer_inplace(layer, bits)
    m.model.embed_tokens = m.model.embed_tokens.to(device)
    for layer in m.model.layers: layer.to(device)
    m.model.norm = m.model.norm.to(device); m.lm_head = m.lm_head.to(device); return m
def state_sha256(m):
    h = hashlib.sha256(); sd = m.state_dict()
    for k in sorted(sd):
        t = sd[k]
        if torch.is_tensor(t) and t.numel(): h.update(k.encode()); h.update(t.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
    return h.hexdigest()
if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="E3"); ap.add_argument("--manifest", default="results/manifests/layer_selection.json"); a = ap.parse_args()
    man = json.load(open(a.manifest)); m = build_arm(a.arm, man); tok = AutoTokenizer.from_pretrained(MID)
    ids = tok(tok.apply_chat_template([{"role": "user", "content": "What is the capital of Japan?"}], tokenize=False, add_generation_prompt=True), return_tensors="pt", add_special_tokens=False).input_ids.to("cuda:0")
    out = m.generate(ids, max_new_tokens=40, do_sample=False, eos_token_id=[3, 6, 261001], pad_token_id=0)
    print("protected layers:", man["unified_matrix"][a.arm]["protected_layers"]); print(tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True)); print("state sha256:", state_sha256(m))
