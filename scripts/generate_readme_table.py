import csv

def generate_markdown_table(csv_path="results/tables/results_table.csv"):
    descriptions = {
        "E0_FP16": "Unquantized FP16 Reference",
        "E1_uniform_W4": "Uniform NF4 4-bit Baseline",
        "S_A_importance_FP16_W4": "Strategy A (Activation Patching)",
        "S_A_importance_W8_W4": "Strategy A (INT8 Protected)",
        "S_B_fragility_FP16_W4": "Strategy B (NF4 KL Sensitivity)",
        "S_B_fragility_W8_W4": "Strategy B (INT8 Protected)",
        "S_C_refusalDrop_FP16_W4": "Strategy C (Refusal Drop)",
        "S_C_refusalDrop_W8_W4": "Strategy C (INT8 Protected)",
        "S_D_composite_FP16_W4": "Strategy D (Composite Score)",
        "S_D_composite_W8_W4": "Strategy D (INT8 Protected)",
        "S_N_negative_FP16_W4": "Negative Control (Low Rank)",
        "S_N_negative_W8_W4": "Negative Control (INT8 Protected)",
        "R0_FP16_W4": "Random Control Set 0",
        "R0_W8_W4": "Random Control Set 0 (INT8)",
        "R1_FP16_W4": "Random Control Set 1",
        "R1_W8_W4": "Random Control Set 1 (INT8)",
        "R2_FP16_W4": "Random Control Set 2",
        "R3_FP16_W4": "Random Control Set 3",
        "S_G_gradattr_FP16_W4": "Strategy G (Gradient Attribution)",
        "S_G_gradattr_W8_W4": "Strategy G (INT8 Protected)",
        "S_X_Cons5_FP16_W4": "Consensus 5-Layer (FP16 Protected)",
        "S_X_Cons5_W8_W4": "Consensus 5-Layer (INT8 Protected)",
        "S_X_C3_W8_W4": "Ablation C3 (3 Layers in INT8)",
        "S_X_C4_W8_W4": "Ablation C4 (4 Layers in INT8)",
        "S_CK_circuitkit_FP16_W4": "CircuitKIT EAP-IG Refusal Circuit",
        "S_CK_circuitkit_W8_W4": "CircuitKIT EAP-IG (INT8 Protected)",
    }

    lines = []
    lines.append("| Arm | Description | GPU Alloc (GB) | Peak (GB) | tok/s | MMLU | Belebele | Harmful Refusal | 95% Bootstrap CI | Benign Refusal (Over-refusal) |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row["arm"]
            desc = descriptions.get(arm, arm)
            gb = f"{float(row['GB']):.2f}"
            peak = f"{float(row['peak_GB']):.2f}"
            tok = f"{float(row['tok_s']):.1f}"
            mmlu = f"{float(row['mmlu']):.3f}"
            bele = f"{float(row['bele']):.3f}"
            ref_h = f"{float(row['refuse_harm']):.3f}"
            ci = f"[{float(row['refuse_harm_lo']):.3f}, {float(row['refuse_harm_hi']):.3f}]"
            ref_b = f"{float(row['refuse_benign']):.3f}"

            is_highlight = arm in ["E0_FP16", "E1_uniform_W4", "S_C_refusalDrop_FP16_W4", "S_CK_circuitkit_FP16_W4", "S_X_Cons5_FP16_W4"]
            arm_fmt = f"**`{arm}`**" if is_highlight else f"`{arm}`"
            desc_fmt = f"**{desc}**" if is_highlight else desc
            ref_fmt = f"**{ref_h}**" if is_highlight else f"{ref_h}"

            lines.append(f"| {arm_fmt} | {desc_fmt} | {gb} | {peak} | {tok} | {mmlu} | {bele} | {ref_fmt} | {ci} | {ref_b} |")

    return "\n".join(lines)

if __name__ == "__main__":
    print(generate_markdown_table())
