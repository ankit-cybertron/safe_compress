# Publication Audit — SafeCompress

> [!NOTE]
> This document categorizes every file type in the workspace by publication safety. It identifies what is safe to publish, what is intentionally withheld, and what requires review before release.

---

## Category 1: Safe to Publish ✅

These files contain no sensitive data, no gated-dataset content, and no credentials.

| File / Directory | Content | Reason Safe |
|---|---|---|
| `safecompress_output/results_dataset/results_table.csv` | Per-arm aggregate metrics (MMLU, Belebele, refuse_harm, refuse_benign, throughput, memory) | Only aggregate scores; no prompt text |
| `safecompress_output/results_dataset/results_comparisons.csv` | Pairwise bootstrap CIs | Derived statistics; no raw prompt data |
| `safecompress_output/results_dataset/auditkit_results.csv` | AuditKIT per-arm scores (13 arms) | Aggregate scores only |
| `safecompress_output/results_dataset/auditkit_comparisons.csv` | AuditKIT pairwise CIs (14 rows) | Derived statistics |
| `safecompress_output/results_dataset/figs/final_results.png` | Results visualization | No sensitive content |
| `safecompress_output/model_repo/final_results.png` | Same figure (duplicate) | No sensitive content |
| `safecompress_output/results_dataset/code/` | Experiment scripts (layer scoring, eval, arm construction, AuditKIT) | Code only; no credentials or prompt data |
| `safecompress_output/model_repo/load_safecompress.py` | Model loader script | Code only |
| `safecompress_output/results_dataset/FINDINGS_layer_scoring.md` | Layer scoring narrative and analysis | Research notes; no sensitive data |
| `safecompress_output/results_dataset/candidates.json` | Layer set definitions | Six named dicts of integer indices |
| `safe-compress.ipynb` | Full experiment notebook | Outputs are scores/figures; no prompt text in outputs |
| `docs/` (all files) | Documentation | Research documentation; no sensitive data |
| `README.md` | Submission overview | No sensitive data |

### HF Published Artifacts (Already Public)

| Artifact | Location | Content |
|---|---|---|
| 20-arm model collection | `ankittiwarifr/safecompress` (HuggingFace) | Quantized model weights |
| Results dataset | `ankittiwarifr/safecompress-results` (HuggingFace) | Scores, figures, code — NOT prompts or replies |

---

## Category 2: Withheld Intentionally 🔒

These files are **not** in the submission because the source datasets are gated or restricted.

| Content Type | Reason Withheld | Source Datasets |
|---|---|---|
| Evaluation prompts (safety prompts) | Sourced from gated datasets (HarmBench-standard, PolyGuardMix) | HarmBench-standard, PolyGuardMix |
| Model replies to prompts | Derived from gated prompt inputs; withheld to respect gating | As above |
| Per-arm raw response JSONL files (if they contain prompt text) | Would expose gated prompt content | As above |

> [!IMPORTANT]
> The dataset published at `ankittiwarifr/safecompress-results` contains scores, figures, and code — it explicitly does **not** contain prompt text or model replies. This is consistent with the terms of the gated source datasets.

---

## Category 3: Requires Review — Per-Arm Result JSONs 🔍

| File Type | Content | Assessment |
|---|---|---|
| Per-arm result JSONs | Each record contains a `prompt_id` field (e.g., `"HarmBench-standard:107"`) plus the model's reply text and score | See below |

**Assessment:** The `prompt_id` field is a **reference index**, not prompt text. The value `"HarmBench-standard:107"` means "prompt number 107 from the HarmBench-standard benchmark." It does not expose the prompt content itself. The reply text, however, is a model generation conditioned on the gated prompt.

**Recommendation:**
- Prompt IDs alone: **SAFE to publish** (they are indices, not content)
- Reply text in per-arm JSONs: **Requires case-by-case review**. If replies are benign (refusals or capability responses to benign prompts), they may be publishable. If replies to harmful prompts contain harmful content, they must be withheld.
- Current submission practice: prompts and replies withheld; only aggregate scores published. This is the **conservative and correct** approach.

---

## Category 4: Credentials and Tokens 🔑

| Item | Status | Detail |
|---|---|---|
| HF_TOKEN | ✅ Safe — not hardcoded | The Hugging Face token is read from an environment variable (`os.environ["HF_TOKEN"]` or Kaggle secrets). It is not present in any script or notebook cell as a literal string. |

> [!CAUTION]
> Before publishing `safe-compress.ipynb`, verify that no notebook output cell contains a printed token value. If the token was ever accidentally printed to stdout (e.g., via a debug statement), that cell output must be cleared before publication.

---

## Publication Checklist

- [x] Aggregate scores: safe and published  
- [x] Figures: safe and published  
- [x] Code and scripts: safe and published  
- [x] Layer definitions (candidates.json): safe and published  
- [x] HF_TOKEN: not hardcoded, safe  
- [ ] Notebook cell outputs: verify no token values printed  
- [ ] Per-arm reply text: withheld (conservative approach maintained)  
- [ ] Prompt text: withheld (gated source datasets)  
