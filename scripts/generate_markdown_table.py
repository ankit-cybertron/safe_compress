import csv

def generate_table(csv_path="results/tables/results_table.csv"):
    arm_descriptions = {
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
        "S_G_gradattr_FP16_W4": "Strategy G (Gradient Attribution)",
        "S_G_gradattr_W8_W4": "Strategy G (INT8 Protected)",
        "S_N_negative_FP16_W4": "Negative Control (Low Rank)",
        "S_N_negative_W8_W4": "Negative Control (INT8 Protected)",
        "R0_FP16_W4": "Random Control Set 0",
        "R0_W8_W4": "Random Control Set 0 (INT8)",
        "R1_FP16_W4": "Random Control Set 1",
        "R1_W8_W4": "Random Control Set 1 (INT8)",
        "R2_FP16_W4": "Random Control Set 2",
        "R3_FP16_W4": "Random Control Set 3",
    }

    headers = [
        "Arm", "Description", "GPU Alloc (GB)", "Peak (GB)", "tok/s",
        "MMLU", "Belebele", "Harmful Refusal", "95% Bootstrap CI", "Benign Refusal (Over-refusal)"
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |"
    ]

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            arm = row["arm"]
            desc = arm_descriptions.get(arm, arm)
            gb = f"{float(row['GB']):.2f}"
            peak_gb = f"{float(row['peak_GB']):.2f}"
            tok_s = f"{float(row['tok_s']):.1f}"
            mmlu = f"{float(row['mmlu']):.3f}"
            bele = f"{float(row['bele']):.3f}"
            refuse = f"{float(row['refuse_harm']):.3f}"
            ref_lo = f"{float(row['refuse_harm_lo']):.3f}"
            ref_hi = f"{float(row['refuse_harm_hi']):.3f}"
            ci = f"[{ref_lo}, {ref_hi}]"
            benign = f"{float(row['refuse_benign']):.3f}"

            # Bold specific key arms
            if arm in ["E0_FP16", "E1_uniform_W4", "S_C_refusalDrop_FP16_W4"]:
                arm_str = f"**`{arm}`**"
                if arm == "S_C_refusalDrop_FP16_W4":
                    desc_str = f"**{desc}**"
                    gb_str = f"**{gb}**"
                    peak_str = f"**{peak_gb}**"
                    tok_str = f"**{tok_s}**"
                    mmlu_str = f"**{mmlu}**"
                    bele_str = f"**{bele}**"
                    refuse_str = f"**{refuse}**"
                    ci_str = f"**{ci}**"
                    benign_str = f"**{benign}**"
                else:
                    desc_str = desc
                    gb_str = gb
                    peak_str = peak_gb
                    tok_str = tok_s
                    mmlu_str = mmlu
                    bele_str = bele
                    refuse_str = f"**{refuse}**"
                    ci_str = ci
                    benign_str = benign
            else:
                arm_str = f"`{arm}`"
                desc_str = desc
                gb_str = gb
                peak_str = peak_gb
                tok_str = tok_s
                mmlu_str = mmlu
                bele_str = bele
                refuse_str = f"**{refuse}**"
                ci_str = ci
                benign_str = benign

            line = f"| {arm_str} | {desc_str} | {gb_str} | {peak_str} | {tok_str} | {mmlu_str} | {bele_str} | {refuse_str} | {ci_str} | {benign_str} |"
            lines.append(line)

    return "\n".join(lines)

if __name__ == "__main__":
    table_md = generate_table()
    print(table_md)
    with open("results/tables/main_table_generated.md", "w") as out:
        out.write(table_md)
