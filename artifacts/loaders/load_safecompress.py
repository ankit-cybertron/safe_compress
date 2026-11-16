import torch, bitsandbytes as bnb
from transformers import AutoModelForCausalLM
def _quant(layer, bits):
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
def load_safecompress(protect_layers, protect_bits=8, model_id="CohereLabs/tiny-aya-global", token=None, device="cuda:0"):
    m = AutoModelForCausalLM.from_pretrained(model_id, token=token, dtype=torch.float16, low_cpu_mem_usage=True).eval()
    for i, ly in enumerate(m.model.layers):
        b = protect_bits if i in set(protect_layers) else 4
        if b < 16: _quant(ly, b)
    m.model.embed_tokens = m.model.embed_tokens.to(device)
    for ly in m.model.layers: ly.to(device)
    m.model.norm = m.model.norm.to(device); m.lm_head = m.lm_head.to(device); return m
