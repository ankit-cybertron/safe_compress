# SafeCompress

Does keeping patching-identified layers at higher precision during quantization preserve refusal, over-refusal and capability (English and Hindi) in CohereLabs/tiny-aya-global better than parameter-matched random layers, least-effect layers and uniform quantization?

**Headline:** Keeping the patching-identified layers at higher precision did NOT help refusal: it was worse than random layers in at least one scheme. Other metrics: over-refusal: no distinguishable difference; capability: no distinguishable difference.

- `results/` per-arm tables, final statistics, per-prompt refusal labels (ids and labels only, no prompt or output text), figures
- `scripts/` patching worker, evaluation workers, loader for any arm
- `ARTICLE.md` full write-up with limitations

Licensing: base model CohereLabs/tiny-aya-global is CC-BY-NC 4.0 with an acceptable-use addendum; this project is non-commercial research. Benchmark prompts and outputs are not redistributed.

## Limitations (short)
- The layer ranking comes from our own activation-patching code (whole-sequence, forward passes, equal-length harmful/unharmful pairs from PolyGuardMix), not from CircuitKIT/EAP. Early layers (layer 0 scores 0.998) show large denoising effects with near-zero noising effects, which suggests the score partly measures prompt-content flow rather than a safety-specific circuit. The 5th and 6th layers score almost the same (0.305 vs 0.279). Treat the set as "patching-identified layers", not a verified refusal circuit.
- Only 3 random layer sets per scheme (the frozen manifest contains 10, only 3 were evaluated). Three controls cannot give a reliable null distribution.
- Refusal is detected with a rule-based phrase matcher on the first 250 characters of a 64-token greedy response, not a judge model. Our first matcher counted words such as "harmful" and "illegal" as refusals; the numbers here use a stricter matcher and we report the original too in final_experimental_matrix.csv.
- The first safety run truncated every prompt at 512 tokens (including a ~365-token default system preamble). Prompts above that length (186 prompts incl. HarmBench-copyright) were removed from the analysis. Length filter applied: True.
- Capability is 500 paired Global-MMLU questions per language; a single arm has a 95% interval of roughly +-4.4 points, so differences under about 3 points are not resolvable. Prompts used a short answer-letter format with a fixed letter-token set.
- One model (Tiny Aya Global), one quantiser family (bitsandbytes NF4 and LLM.int8), greedy decoding, English and Hindi only. No 3-bit arm and no GPTQ/AWQ/HQQ backend. No human review of outputs.
- Safety benchmark prompts and model outputs on them are not redistributed. Only ids, labels and numbers are in this repository.
