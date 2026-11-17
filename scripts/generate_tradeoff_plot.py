import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_tradeoff_plot(csv_path="results/tables/results_table.csv", output_paths=None):
    if output_paths is None:
        output_paths = [
            "figures/tradeoffs/safety_capability_tradeoff_26arms.png",
        ]

    arms_data = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            arm_id = row["arm"]
            mmlu = float(row["mmlu"]) * 100.0
            refusal = float(row["refuse_harm"]) * 100.0
            arms_data.append((i + 1, arm_id, mmlu, refusal))

    # Display labels for legend
    legend_labels = {
        "E0_FP16": "FP16 baseline",
        "E1_uniform_W4": "Uniform W4",
        "S_A_importance_FP16_W4": "A importance (FP16/W4)",
        "S_A_importance_W8_W4": "A importance (W8/W4)",
        "S_B_fragility_FP16_W4": "B fragility (FP16/W4)",
        "S_B_fragility_W8_W4": "B fragility (W8/W4)",
        "S_C_refusalDrop_FP16_W4": "C refusal-drop (FP16/W4)",
        "S_C_refusalDrop_W8_W4": "C refusal-drop (W8/W4)",
        "S_D_composite_FP16_W4": "D composite (FP16/W4)",
        "S_D_composite_W8_W4": "D composite (W8/W4)",
        "S_N_negative_FP16_W4": "Negative control (FP16/W4)",
        "S_N_negative_W8_W4": "Negative control (W8/W4)",
        "R0_FP16_W4": "Random 0 (FP16/W4)",
        "R0_W8_W4": "Random 0 (W8/W4)",
        "R1_FP16_W4": "Random 1 (FP16/W4)",
        "R1_W8_W4": "Random 1 (W8/W4)",
        "R2_FP16_W4": "Random 2 (FP16/W4)",
        "R3_FP16_W4": "Random 3 (FP16/W4)",
        "S_G_gradattr_FP16_W4": "G grad-attr (FP16/W4)",
        "S_G_gradattr_W8_W4": "G grad-attr (W8/W4)",
        "S_X_Cons5_FP16_W4": "Consensus 5 (FP16/W4)",
        "S_X_Cons5_W8_W4": "Consensus 5 (W8/W4)",
        "S_X_C3_W8_W4": "Ablation C3 (W8/W4)",
        "S_X_C4_W8_W4": "Ablation C4 (W8/W4)",
        "S_CK_circuitkit_FP16_W4": "CircuitKIT EAP-IG (FP16/W4)",
        "S_CK_circuitkit_W8_W4": "CircuitKIT EAP-IG (W8/W4)",
    }

    # Distinct styles for 26 arms
    styles = [
        ("o", "#1f77b4"),  # 01 E0
        ("s", "#ff7f0e"),  # 02 E1
        ("D", "#2ca02c"),  # 03 A FP16
        ("d", "#d62728"),  # 04 A W8
        ("P", "#9467bd"),  # 05 B FP16
        ("X", "#8c564b"),  # 06 B W8
        ("o", "#e377c2"),  # 07 C FP16
        ("s", "#7f7f7f"),  # 08 C W8
        ("^", "#bcbd22"),  # 09 D FP16
        ("D", "#17becf"),  # 10 D W8
        ("P", "#1f77b4"),  # 11 N FP16
        ("X", "#ff7f0e"),  # 12 N W8
        ("o", "#2ca02c"),  # 13 R0 FP16
        ("s", "#d62728"),  # 14 R0 W8
        ("^", "#9467bd"),  # 15 R1 FP16
        ("v", "#8c564b"),  # 16 R1 W8
        ("v", "#e377c2"),  # 17 R2 FP16
        ("X", "#7f7f7f"),  # 18 R3 FP16
        ("p", "#b8860b"),  # 19 G FP16
        ("*", "#4b0082"),  # 20 G W8
        ("h", "#008080"),  # 21 Cons5 FP16
        ("H", "#ff1493"),  # 22 Cons5 W8
        ("<", "#00bfff"),  # 23 C3 W8
        (">", "#ff4500"),  # 24 C4 W8
        ("o", "#2e8b57"),  # 25 CK FP16
        ("D", "#8a2be2"),  # 26 CK W8
    ]

    fig, ax = plt.subplots(figsize=(11, 8.5), dpi=300, facecolor="#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Grid
    ax.grid(True, linestyle=":", alpha=0.5, color="#CCCCCC")
    ax.set_axisbelow(True)

    legend_entries = []

    # Manual offsets to prevent overlaps
    offsets = {
        1: (8, 5),
        2: (-15, 6),
        3: (8, -6),
        4: (8, 4),
        5: (8, 4),
        6: (-15, 4),
        7: (8, 5),
        8: (8, 4),
        9: (8, 4),
        10: (8, 4),
        11: (8, 4),
        12: (8, 4),
        13: (8, -6),
        14: (8, 4),
        15: (8, 4),
        16: (-16, -6),
        17: (8, 4),
        18: (8, 4),
        19: (8, 4),
        20: (-16, 4),
        21: (8, 4),
        22: (-16, 4),
        23: (8, -5),
        24: (8, 4),
        25: (8, 5),
        26: (-16, 5),
    }

    for num, arm, mmlu, refusal in arms_data:
        marker, color = styles[num - 1]
        ax.scatter(mmlu, refusal, marker=marker, color=color, s=85, edgecolors="black", linewidths=0.6, zorder=4)

        ox, oy = offsets.get(num, (8, 4))
        ax.annotate(
            f"{num}",
            (mmlu, refusal),
            xytext=(ox, oy),
            textcoords="offset points",
            fontsize=7.5,
            fontweight="bold",
            bbox=dict(boxstyle="circle,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.9),
            zorder=5,
        )

        display_name = legend_labels.get(arm, arm)
        legend_entries.append(f"{num:02d}. {display_name}")

    ax.set_title("Safety-capability trade-off across 26 experimental arms", fontsize=13, fontweight="bold", pad=12, loc="center")
    ax.set_xlabel("Global-MMLU accuracy (%)", fontsize=10.5, labelpad=8)
    ax.set_ylabel("Harmful-prompt refusal rate (%)", fontsize=10.5, labelpad=8)

    ax.set_xlim(45.0, 48.2)
    ax.set_ylim(60.0, 66.8)

    legend_text = "\n".join(legend_entries)
    ax.text(
        1.03,
        0.5,
        legend_text,
        transform=ax.transAxes,
        fontsize=7.2,
        verticalalignment="center",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#F8FAFC", edgecolor="#CBD5E1", lw=0.8),
    )

    fig.text(
        0.12,
        0.02,
        "Each point is one configuration. Numbered labels map to the configuration key on the right.",
        fontsize=7.5,
        color="#64748B",
    )

    plt.subplots_adjust(right=0.70, bottom=0.1)
    for p in output_paths:
        fig.savefig(p, dpi=300, facecolor="#FFFFFF", bbox_inches="tight")
        print(f"Generated tradeoff plot at: {p}")
    plt.close(fig)

if __name__ == "__main__":
    generate_tradeoff_plot()
