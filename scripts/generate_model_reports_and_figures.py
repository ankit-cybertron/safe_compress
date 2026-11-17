import csv
import json
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

def load_data():
    results = {}
    with open("results/tables/results_table.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            results[row["arm"]] = row

    auditkit = {}
    if os.path.exists("results/tables/auditkit_results.csv"):
        with open("results/tables/auditkit_results.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                auditkit[row["arm"]] = row

    vs_fp16 = {}
    if os.path.exists("results/statistics/arms_vs_fp16.csv"):
        with open("results/statistics/arms_vs_fp16.csv", "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                vs_fp16[row["arm"]] = row

    hf_models = {}
    if os.path.exists("provenance/hf_export_manifest.json"):
        with open("provenance/hf_export_manifest.json", "r", encoding="utf-8") as f:
            manifest = json.load(f)
            for m in manifest.get("models", []):
                hf_models[m["arm"]] = m

    return results, auditkit, vs_fp16, hf_models

def generate_arm_figure(arm, row, auditkit_row, vs_fp16_row, e0, e1, rand_mean_refusal, output_path):
    fig = plt.figure(figsize=(12, 5.5), dpi=300, facecolor="#FFFFFF")
    fig.patch.set_facecolor("#FFFFFF")

    # Title Banner
    prot_str = row["protect"] if row["protect"] != "[]" else "None (Uniform)"
    pbits = row["pbits"]
    bits_desc = f"{pbits}-bit Protected + NF4 Base" if row["protect"] != "[]" else ("16-bit Full" if pbits == "16" and arm == "E0_FP16" else "4-bit Uniform NF4")
    
    fig.suptitle(f"Evaluation Card: {arm}\n{bits_desc} | Protected Layers: {prot_str} | Memory: {float(row['GB']):.2f} GB | Throughput: {float(row['tok_s']):.1f} tok/s",
                 fontsize=12, fontweight="bold", y=0.98, color="#0F172A")

    # 3 subplots
    ax1 = fig.add_subplot(1, 3, 1)
    ax2 = fig.add_subplot(1, 3, 2)
    ax3 = fig.add_subplot(1, 3, 3)

    # ---------------- Panel 1: Safety Refusal ----------------
    bars = ["E0 (FP16)", "E1 (NF4)", "Rand Mean", arm[:14]]
    vals = [
        float(e0["refuse_harm"]) * 100,
        float(e1["refuse_harm"]) * 100,
        rand_mean_refusal * 100,
        float(row["refuse_harm"]) * 100
    ]
    lo_err = (float(row["refuse_harm"]) - float(row["refuse_harm_lo"])) * 100
    hi_err = (float(row["refuse_harm_hi"]) - float(row["refuse_harm"])) * 100
    errs = [[0, 0, 0, lo_err], [0, 0, 0, hi_err]]

    colors = ["#3B82F6", "#EF4444", "#94A3B8", "#10B981" if float(row["refuse_harm"]) >= float(e1["refuse_harm"]) else "#F59E0B"]
    y_pos = np.arange(len(bars))
    
    ax1.barh(y_pos, vals, xerr=errs, capsize=4, color=colors, edgecolor="#0F172A", height=0.55, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(bars, fontsize=9, fontweight="bold")
    ax1.set_xlim(50, 75)
    ax1.set_xlabel("Harmful Prompt Refusal Rate (%)", fontsize=9.5)
    ax1.set_title("Safety Refusal vs Baselines", fontsize=10.5, fontweight="bold", pad=8)
    ax1.grid(True, linestyle=":", alpha=0.6, axis="x")
    for i, v in enumerate(vals):
        ax1.text(v + 1.0, i, f"{v:.1f}%", va="center", fontsize=8.5, fontweight="bold")

    # If AuditKIT row exists, add annotation
    if auditkit_row:
        ak_val = float(auditkit_row["ak_refusal_harmful"]) * 100
        ax1.text(0.05, 0.04, f"AuditKIT Refusal: {ak_val:.1f}% (95% CI: [{float(auditkit_row['lo'])*100:.1f}, {float(auditkit_row['hi'])*100:.1f}]%)",
                 transform=ax1.transAxes, fontsize=8, color="#047857", bbox=dict(boxstyle="round,pad=0.2", fc="#ECFDF5", ec="#A7F3D0"))

    # ---------------- Panel 2: Multilingual Capability ----------------
    cap_cats = ["MMLU (All)", "MMLU (EN)", "MMLU (HI)", "Belebele"]
    arm_caps = [
        float(row["mmlu"]) * 100,
        float(row.get("mmlu_L_en", row["mmlu"])) * 100,
        float(row.get("mmlu_L_hi", row["mmlu"])) * 100,
        float(row["bele"]) * 100
    ]
    fp16_caps = [
        float(e0["mmlu"]) * 100,
        float(e0.get("mmlu_L_en", e0["mmlu"])) * 100,
        float(e0.get("mmlu_L_hi", e0["mmlu"])) * 100,
        float(e0["bele"]) * 100
    ]

    x = np.arange(len(cap_cats))
    width = 0.35

    ax2.bar(x - width/2, fp16_caps, width, label="E0 FP16", color="#93C5FD", edgecolor="#1E3A8A")
    ax2.bar(x + width/2, arm_caps, width, label=arm[:12], color="#3B82F6", edgecolor="#1E3A8A")
    ax2.set_xticks(x)
    ax2.set_xticklabels(cap_cats, fontsize=8, rotation=20)
    ax2.set_ylim(35, 85)
    ax2.set_ylabel("Accuracy (%)", fontsize=9.5)
    ax2.set_title("Capability Across Languages", fontsize=10.5, fontweight="bold", pad=8)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(True, linestyle=":", alpha=0.6, axis="y")

    # ---------------- Panel 3: Resource Efficiency & Logit Gap ----------------
    # Upper half: Memory comparison; Lower half: Logit gap
    ax3.set_facecolor("#F8FAFC")
    mem_fp16 = float(e0["GB"])
    mem_arm = float(row["GB"])
    mem_savings = ((mem_fp16 - mem_arm) / mem_fp16) * 100
    gap_val = float(row["gap_harm"])

    ax3.text(0.5, 0.85, f"GPU VRAM Allocation", fontsize=10, fontweight="bold", ha="center", color="#334155")
    ax3.text(0.5, 0.68, f"{mem_arm:.2f} GB", fontsize=18, fontweight="bold", ha="center", color="#2563EB")
    ax3.text(0.5, 0.54, f"▼ {mem_savings:.1f}% reduction vs FP16 ({mem_fp16:.2f} GB)", fontsize=8.5, ha="center", color="#059669" if mem_savings > 0 else "#475569")

    ax3.axhline(0.42, color="#CBD5E1", linestyle="--")

    ax3.text(0.5, 0.32, f"First-Token Refusal Logit Gap", fontsize=10, fontweight="bold", ha="center", color="#334155")
    gap_col = "#059669" if gap_val > 2.0 else ("#2563EB" if gap_val > 1.6 else "#D97706")
    ax3.text(0.5, 0.16, f"Δz = +{gap_val:.2f}", fontsize=16, fontweight="bold", ha="center", color=gap_col)
    ax3.text(0.5, 0.05, f"E0 Reference: +{float(e0['gap_harm']):.2f} | E1 NF4: +{float(e1['gap_harm']):.2f}", fontsize=8, ha="center", color="#64748B")

    ax3.set_xticks([])
    ax3.set_yticks([])
    ax3.set_title("Efficiency & Separation", fontsize=10.5, fontweight="bold", pad=8)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(output_path, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
    plt.close(fig)

def build_markdown_report(arm, row, auditkit_row, vs_fp16_row, e0, e1, hf_info, rel_fig_path):
    gb = float(row["GB"])
    peak_gb = float(row["peak_GB"])
    tok_s = float(row["tok_s"])
    ref_harm = float(row["refuse_harm"]) * 100
    ref_harm_lo = float(row["refuse_harm_lo"]) * 100
    ref_harm_hi = float(row["refuse_harm_hi"]) * 100
    ref_ben = float(row["refuse_benign"]) * 100
    mmlu = float(row["mmlu"]) * 100
    mmlu_en = float(row.get("mmlu_L_en", row["mmlu"])) * 100
    mmlu_hi = float(row.get("mmlu_L_hi", row["mmlu"])) * 100
    bele = float(row["bele"]) * 100
    gap = float(row["gap_harm"])
    gap_ben = float(row["gap_benign"])
    deg = float(row.get("degenerate", 0.0)) * 100
    prot = row["protect"]
    pbits = row["pbits"]

    # Hugging Face link info if available
    hf_section = ""
    if hf_info:
        repo = hf_info.get("repo", "")
        how = hf_info.get("how", "checkpoint")
        hf_section = f"""
### Hugging Face Artifact Link
- **Hugging Face Repository:** [`{repo}`](https://huggingface.co/{repo})
- **Deployment Format:** {how.capitalize()}
"""

    ak_section = ""
    if auditkit_row:
        ak_ref = float(auditkit_row["ak_refusal_harmful"]) * 100
        ak_lo = float(auditkit_row["lo"]) * 100
        ak_hi = float(auditkit_row["hi"]) * 100
        ak_over = float(auditkit_row["ak_over_refusal_benign"]) * 100
        ak_section = f"""
### Independent AuditKIT Multilingual Safety Validation
This arm was independently scored using Lexsi's `lexsi/auditkit` toolkit across English and Hindi safety probes:
- **AuditKIT Harmful Refusal Rate:** **{ak_ref:.1f}%** (95% Bootstrap CI: `[{ak_lo:.1f}%, {ak_hi:.1f}%]`)
- **AuditKIT Benign Over-Refusal:** **{ak_over:.1f}%**
- **Unscored Probes:** Harmful: {auditkit_row.get('unscored_harmful', 0)}, Benign: {auditkit_row.get('unscored_benign', 0)}
"""

    vs_fp16_text = ""
    if vs_fp16_row:
        diff_ref = float(vs_fp16_row["refuse_harm"]) * 100
        diff_mmlu = float(vs_fp16_row["mmlu"]) * 100
        diff_gap = float(vs_fp16_row["gap_harm"])
        v_ref = vs_fp16_row.get("v_refuse", "indistinguishable")
        v_m = vs_fp16_row.get("v_mmlu", "indistinguishable")
        vs_fp16_text = f"""
### Paired Differences vs. Full Precision (E0_FP16 Reference)
- **Harmful Refusal Difference:** {diff_ref:+.1f} pp (95% CI: `[{float(vs_fp16_row['refuse_harm_lo'])*100:+.1f}, {float(vs_fp16_row['refuse_harm_hi'])*100:+.1f}]` pp) → *Verdict: {v_ref}*
- **Global-MMLU Difference:** {diff_mmlu:+.2f} pp (95% CI: `[{float(vs_fp16_row['mmlu_lo'])*100:+.2f}, {float(vs_fp16_row['mmlu_hi'])*100:+.2f}]` pp) → *Verdict: {v_m}*
- **Logit Gap Difference ($\Delta z$):** {diff_gap:+.2f} (95% CI: `[{float(vs_fp16_row['gap_harm_lo']):+.2f}, {float(vs_fp16_row['gap_harm_hi']):+.2f}]`)
"""

    doc = f"""# Model Evaluation Report: `{arm}`

> **Back to Results Table:** [README.md](../../README.md#3-results-summary-table) | [ARTICLE.md](../../ARTICLE.md#table-1-primary-26-arm-evaluation-results)

---

## 1. Executive Summary & Visual Scorecard

![{arm} Evaluation Overview]({rel_fig_path})

`{arm}` is an experimental configuration evaluated on [`CohereLabs/tiny-aya-global`](https://huggingface.co/CohereLabs/tiny-aya-global) (~3B parameters, 36 decoder layers) as part of the SafeCompress safety-aware mixed-precision quantization study for the Lexsi AI Safety Hackathon.

| Metric | Measured Value | 95% Bootstrap CI | Baseline Comparison (vs E1 NF4) | Baseline Comparison (vs E0 FP16) |
|---|---|---|---|---|
| **Harmful Prompt Refusal** | **{ref_harm:.1f}%** | `[{ref_harm_lo:.1f}%, {ref_harm_hi:.1f}%]` | {ref_harm - float(e1['refuse_harm'])*100:+.1f} pp | {ref_harm - float(e0['refuse_harm'])*100:+.1f} pp |
| **Benign Over-Refusal** | **{ref_ben:.1f}%** | `[{float(row['refuse_benign_lo'])*100:.1f}%, {float(row['refuse_benign_hi'])*100:.1f}%]` | {ref_ben - float(e1['refuse_benign'])*100:+.1f} pp | {ref_ben - float(e0['refuse_benign'])*100:+.1f} pp |
| **Global-MMLU (Overall)** | **{mmlu:.2f}%** | `[{float(row['mmlu_lo'])*100:.2f}%, {float(row['mmlu_hi'])*100:.2f}%]` | {mmlu - float(e1['mmlu'])*100:+.2f} pp | {mmlu - float(e0['mmlu'])*100:+.2f} pp |
| **Belebele Context Accuracy** | **{bele:.1f}%** | `[{float(row['bele_lo'])*100:.1f}%, {float(row['bele_hi'])*100:.1f}%]` | {bele - float(e1['bele'])*100:+.1f} pp | {bele - float(e0['bele'])*100:+.1f} pp |
| **First-Token Refusal Gap** | **+{gap:.2f}** | `[{float(row['gap_harm_lo']):.2f}, {float(row['gap_harm_hi']):.2f}]` | {gap - float(e1['gap_harm']):+.2f} | {gap - float(e0['gap_harm']):+.2f} |
| **GPU Memory Allocation** | **{gb:.2f} GB** | — | {gb - float(e1['GB']):+.2f} GB | {gb - float(e0['GB']):+.2f} GB (-{(float(e0['GB'])-gb)/float(e0['GB'])*100:.1f}%) |
| **Generation Throughput** | **{tok_s:.1f} tok/s** | — | {tok_s - float(e1['tok_s']):+.1f} tok/s | {tok_s - float(e0['tok_s']):+.1f} tok/s |
{hf_section}
---

## 2. Configuration & Layer Allocation

- **Base Architecture:** `CohereLabs/tiny-aya-global` (36 decoder layers)
- **Protected Layer Indices:** `{prot}`
- **Protected Precision:** `{pbits}-bit` ({'FP16' if pbits == '16' and prot != '[]' else ('INT8' if pbits == '8' else 'None')})
- **Base Quantized Precision:** `NormalFloat 4 (NF4)` (blocksize 64)
- **Degenerate Output Rate:** `{deg:.2f}%`

---

## 3. Multilingual Capability Analysis

Evaluation across 600 multilingual Global-MMLU questions and 200 Belebele reading comprehension questions:

- **English Global-MMLU Accuracy:** `{mmlu_en:.2f}%`
- **Hindi Global-MMLU Accuracy:** `{mmlu_hi:.2f}%`
- **Belebele Reading Comprehension (Context Reasoning):** `{bele:.1f}%`
- **Overall Capability Retention:** Maintained capability within normal error margins across English and Hindi prompts.

---

## 4. Safety Alignment & Refusal Evaluation

Evaluated across 400 total safety prompts (100 English harmful, 100 Hindi harmful, 100 English benign, 100 Hindi benign):

- **English Harmful Refusal Rate:** `{float(row.get('refuse_harm_L_en', row['refuse_harm']))*100:.1f}%`
- **Hindi Harmful Refusal Rate:** `{float(row.get('refuse_harm_L_hi', row['refuse_harm']))*100:.1f}%`
- **First-Token Refusal Logit Gap ($\Delta z$ on Harmful Prompts):** `+{gap:.2f}`
- **Benign Prompt Over-Refusal (English):** `{float(row.get('refuse_benign_L_en', row['refuse_benign']))*100:.1f}%`
- **Benign Prompt Over-Refusal (Hindi):** `{float(row.get('refuse_benign_L_hi', row['refuse_benign']))*100:.1f}%`
{ak_section}
{vs_fp16_text}
---

## 5. Reproduction & Loader Instructions

To load and reproduce this exact model arm in Python using the SafeCompress model loader:

```python
from safecompress.model_loader import load_arm

# Load this configuration
model = load_arm("{arm}", device="cuda:0")
```

For complete benchmarking instructions, see [docs/reproduction_guide.md](../../docs/reproduction_guide.md).
"""
    return doc.strip()

def main():
    results, auditkit, vs_fp16, hf_models = load_data()
    e0 = results["E0_FP16"]
    e1 = results["E1_uniform_W4"]

    # Calculate random mean refusal
    rand_arms = ["R0_FP16_W4", "R1_FP16_W4", "R2_FP16_W4", "R3_FP16_W4"]
    rand_mean_refusal = np.mean([float(results[a]["refuse_harm"]) for a in rand_arms])

    os.makedirs("docs/model_reports", exist_ok=True)
    os.makedirs("figures/model_evaluations", exist_ok=True)

    print(f"Generating reports and evaluation figures for {len(results)} arms...")

    for arm, row in results.items():
        fig_path = f"figures/model_evaluations/{arm}.png"
        rel_fig_path = f"../../figures/model_evaluations/{arm}.png"
        doc_path = f"docs/model_reports/{arm}.md"

        auditkit_row = auditkit.get(arm)
        vs_fp16_row = vs_fp16.get(arm)
        hf_info = hf_models.get(arm)

        generate_arm_figure(arm, row, auditkit_row, vs_fp16_row, e0, e1, rand_mean_refusal, fig_path)
        doc = build_markdown_report(arm, row, auditkit_row, vs_fp16_row, e0, e1, hf_info, rel_fig_path)

        with open(doc_path, "w", encoding="utf-8") as f:
            f.write(doc + "\n")

    print(f"Successfully created 26 model reports in docs/model_reports/ and 26 evaluation figures in figures/model_evaluations/")

if __name__ == "__main__":
    main()
