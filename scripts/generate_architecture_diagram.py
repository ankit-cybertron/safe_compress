import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ArrowStyle

def create_architecture_diagram(output_path="figures/methodology/safecompress_architecture_diagram.png"):
    # High resolution, wide layout
    fig = plt.figure(figsize=(19, 10.5), dpi=300, facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 19)
    ax.set_ylim(0, 10.5)
    ax.axis("off")

    # Header / Title Block
    title_box = FancyBboxPatch((0.5, 9.3), 18.0, 1.0, boxstyle="round,pad=0.1,rounding_size=0.15",
                               ec="#CBD5E1", fc="#F8FAFC", linewidth=1.5)
    ax.add_patch(title_box)
    ax.text(9.5, 9.9, "SafeCompress: Layer-Informed Mixed-Precision Quantization Architecture",
            fontsize=20, fontweight="bold", ha="center", va="center", color="#0F172A", fontfamily="sans-serif")
    ax.text(9.5, 9.5, "Preserving Safety-Critical Refusal Under 4-Bit Compression via Mechanistic Interpretability & Sensitivity Analysis",
            fontsize=12, ha="center", va="center", color="#475569", fontfamily="sans-serif")

    # Colors
    c_blue = "#2563EB"
    c_teal = "#0D9488"
    c_amber = "#D97706"
    c_purple = "#7C3AED"
    c_slate = "#334155"
    c_light_bg = "#FFFFFF"
    c_border = "#E2E8F0"

    # Stage Box Dimensions
    col_w = 4.1
    col_h = 8.1
    y_top = 8.9
    y_box = y_top - col_h

    columns = [
        {"x": 0.5, "title": "1. BASE MODEL & PROFILING", "sub": "36-Layer Mechanistic & Sensitivity Scan", "color": "#1E40AF", "border": "#BFDBFE", "header_bg": "#EFF6FF"},
        {"x": 5.0, "title": "2. LAYER SELECTION ENGINE", "sub": "Targeted K=5 Layer Allocation", "color": "#0F766E", "border": "#99F6E4", "header_bg": "#F0FDFA"},
        {"x": 9.5, "title": "3. MIXED-PRECISION CORE", "sub": "Asymmetric FP16/INT8 + NF4 Quantization", "color": "#6D28D9", "border": "#DDD6FE", "header_bg": "#F5F3FF"},
        {"x": 14.0, "title": "4. EVALUATION & VALIDATION", "sub": "Dual-Scorer Benchmark & AuditKIT", "color": "#B45309", "border": "#FDE68A", "header_bg": "#FFFBEB"}
    ]

    for col in columns:
        # Outer Card
        card = FancyBboxPatch((col["x"], y_box), col_w, col_h, boxstyle="round,pad=0.15,rounding_size=0.2",
                              ec=col["border"], fc="#FFFFFF", linewidth=2.0)
        ax.add_patch(card)
        # Header inside card
        header = FancyBboxPatch((col["x"] + 0.05, y_top - 0.95), col_w - 0.1, 0.9, boxstyle="round,pad=0.08,rounding_size=0.15",
                                ec="none", fc=col["header_bg"])
        ax.add_patch(header)
        ax.text(col["x"] + col_w / 2.0, y_top - 0.38, col["title"],
                fontsize=11.5, fontweight="bold", ha="center", va="center", color=col["color"], fontfamily="sans-serif")
        ax.text(col["x"] + col_w / 2.0, y_top - 0.70, col["sub"],
                fontsize=9.0, ha="center", va="center", color="#64748B", fontfamily="sans-serif")

    # -------------------------------------------------------------
    # STAGE 1: BASE MODEL & PROFILING CONTENT
    # -------------------------------------------------------------
    # Base Model Pill
    bm_box = FancyBboxPatch((0.7, 7.0), 3.7, 0.75, boxstyle="round,pad=0.08,rounding_size=0.12",
                            ec="#93C5FD", fc="#F0F7FF", linewidth=1.2)
    ax.add_patch(bm_box)
    ax.text(2.55, 7.48, "CohereLabs/tiny-aya-global (~3B)", fontsize=11, fontweight="bold", ha="center", va="center", color="#1E3A8A")
    ax.text(2.55, 7.20, "36 Layers: 27 Sliding Attention + 9 Full Global Attention", fontsize=8.2, ha="center", va="center", color="#3B82F6")

    # Layer Stack Diagram
    stack_box = FancyBboxPatch((0.7, 5.0), 3.7, 1.8, boxstyle="round,pad=0.08,rounding_size=0.12",
                               ec="#E2E8F0", fc="#F8FAFC", linewidth=1.0)
    ax.add_patch(stack_box)
    ax.text(2.55, 6.62, "Layer Topology (36 Decoder Layers)", fontsize=9.5, fontweight="bold", ha="center", va="center", color="#334155")
    
    # Mini representation of 36 layers
    for l_idx in range(36):
        row = l_idx // 9
        pos = l_idx % 9
        lx = 0.85 + pos * 0.38
        ly = 6.25 - row * 0.38
        is_full = l_idx in [3, 7, 11, 15, 19, 23, 27, 31, 35]
        bg_col = "#60A5FA" if is_full else "#CBD5E1"
        b_box = FancyBboxPatch((lx, ly), 0.32, 0.28, boxstyle="round,pad=0.02,rounding_size=0.04",
                               ec="#94A3B8" if not is_full else "#1D4ED8", fc=bg_col, linewidth=0.8)
        ax.add_patch(b_box)
        ax.text(lx + 0.16, ly + 0.14, str(l_idx), fontsize=6.0, ha="center", va="center",
                color="#FFFFFF" if is_full else "#1E293B", fontweight="bold")
    ax.text(2.55, 5.18, "Blue: Full Global Attention (9) | Grey: Sliding Attention (27)", fontsize=7.5, ha="center", va="center", color="#64748B")

    # 4 Profiling Signals
    p_signals = [
        ("CircuitKIT EAP-IG Attribution", "Integrated Gradients Edge Attribution\nRefusal circuit identification on GPU (peak 24 GB)", "#FEF3C7", "#B45309"),
        ("Causal Activation Patching", "Denoising clean vs corrupted refusal pairs\nScore: patch_block effect on logit gap", "#DBEAFE", "#1E40AF"),
        ("Single-Layer NF4 Fragility", "One-layer round-trip NF4 quantization\nScore: Output logit drift KL divergence", "#EDE9FE", "#6D28D9"),
        ("Refusal Gap Erosion", "Direct drop in refusal vs compliance logits\nScore: Delta(logsumexp P(refusal))", "#CCFBF1", "#0F766E"),
    ]
    for i, (p_t, p_d, p_bg, p_fg) in enumerate(p_signals):
        py = 3.85 - i * 1.02
        p_card = FancyBboxPatch((0.7, py), 3.7, 0.92, boxstyle="round,pad=0.06,rounding_size=0.1",
                                ec="#CBD5E1", fc="#FFFFFF", linewidth=1.0)
        ax.add_patch(p_card)
        tag = FancyBboxPatch((0.85, py + 0.58), 3.4, 0.28, boxstyle="round,pad=0.04,rounding_size=0.06",
                             ec="none", fc=p_bg)
        ax.add_patch(tag)
        ax.text(2.55, py + 0.72, p_t, fontsize=8.5, fontweight="bold", ha="center", va="center", color=p_fg)
        ax.text(2.55, py + 0.28, p_d, fontsize=7.2, ha="center", va="center", color="#475569")

    # -------------------------------------------------------------
    # STAGE 2: LAYER SELECTION ENGINE CONTENT
    # -------------------------------------------------------------
    bud_box = FancyBboxPatch((5.2, 7.0), 3.7, 0.75, boxstyle="round,pad=0.08,rounding_size=0.12",
                             ec="#99F6E4", fc="#F0FDFA", linewidth=1.2)
    ax.add_patch(bud_box)
    ax.text(7.05, 7.48, "Strict Layer Budget: K = 3 to 5 Layers", fontsize=11, fontweight="bold", ha="center", va="center", color="#0F766E")
    ax.text(7.05, 7.20, "13.9% of Model Parameters (5 Layers) in Higher Precision", fontsize=8.5, ha="center", va="center", color="#14B8A6")

    strat_list = [
        ("CircuitKIT Discovery", "[0, 14, 16, 20, 35]", "EAP-IG Attribution: 66.0% Refusal (+5.5 pp)\nTop MMLU: 47.7% (Outperforms FP16 baseline!)", "#D97706", "#FEF3C7", "#B45309", True),
        ("Strategy C: Refusal Drop", "[7, 14, 16, 27, 35]", "Erosion under NF4: 65.5% Refusal (+5.0 pp)\n60% Global Attention Layers (7, 27, 35)", "#14B8A6", "#CCFBF1", "#0F766E", True),
        ("5-Layer Consensus Set", "[0, 7, 14, 16, 35]", "Intersection across methods: 66.0% Refusal\nMaximizes both safety and reasoning preservation", "#0D9488", "#E0F2FE", "#0284C7", False),
        ("Strategy B: Fragility KL", "[0, 7, 12, 16, 35]", "Highest KL Drift Under NF4: 64.5% Refusal\nProtects Input Layer 0 & Output Layer 35", "#38BDF8", "#E0F2FE", "#0369A1", False),
        ("Strategy A: Patching", "[0, 5, 17, 19, 22]", "Causal Activation Patching: 63.0% Refusal\nStrong Clean Signal, Lower Quantization Fit", "#818CF8", "#EEF2FF", "#4338CA", False),
        ("Negative & Random (x4)", "Negative + 4x Random", "Negative Control: 64.0% | Random Mean: 61.3%\nDistinguishes targeted selection from raw capacity", "#94A3B8", "#F1F5F9", "#475569", False)
    ]

    for i, (s_name, s_layers, s_desc, s_color, s_bg, s_fg, is_lead) in enumerate(strat_list):
        sy = 5.95 - i * 0.98
        s_card = FancyBboxPatch((5.2, sy), 3.7, 0.88, boxstyle="round,pad=0.06,rounding_size=0.1",
                                ec=s_color if is_lead else "#E2E8F0", fc="#FFFFFF" if not is_lead else "#F0FDFA", linewidth=1.8 if is_lead else 1.0)
        ax.add_patch(s_card)
        badge = FancyBboxPatch((5.35, sy + 0.52), 1.9, 0.28, boxstyle="round,pad=0.03,rounding_size=0.06",
                               ec="none", fc=s_bg)
        ax.add_patch(badge)
        ax.text(6.3, sy + 0.66, s_name, fontsize=7.8, fontweight="bold", ha="center", va="center", color=s_fg)
        ax.text(8.0, sy + 0.66, s_layers, fontsize=7.8, fontweight="bold", ha="center", va="center", color="#0F172A")
        ax.text(7.05, sy + 0.24, s_desc, fontsize=7.2, ha="center", va="center", color="#475569")

    # -------------------------------------------------------------
    # STAGE 3: MIXED-PRECISION CORE CONTENT
    # -------------------------------------------------------------
    quant_head = FancyBboxPatch((9.7, 7.0), 3.7, 0.75, boxstyle="round,pad=0.08,rounding_size=0.12",
                                ec="#DDD6FE", fc="#F5F3FF", linewidth=1.2)
    ax.add_patch(quant_head)
    ax.text(11.55, 7.48, "Asymmetric Precision Allocation", fontsize=11, fontweight="bold", ha="center", va="center", color="#6D28D9")
    ax.text(11.55, 7.20, "In-Place bitsandbytes Linear Layer Construction", fontsize=8.5, ha="center", va="center", color="#8B5CF6")

    # Visual Partition Box
    part_box = FancyBboxPatch((9.7, 4.45), 3.7, 2.35, boxstyle="round,pad=0.08,rounding_size=0.12",
                              ec="#CBD5E1", fc="#F8FAFC", linewidth=1.0)
    ax.add_patch(part_box)
    ax.text(11.55, 6.55, "Model Memory Distribution (Tiny Aya Global ~3B)", fontsize=9.5, fontweight="bold", ha="center", va="center", color="#1E293B")

    # Top Slice: Protected Layers
    prot_slice = FancyBboxPatch((9.9, 5.5), 3.3, 0.85, boxstyle="round,pad=0.06,rounding_size=0.08",
                                ec="#7C3AED", fc="#EDE9FE", linewidth=1.5)
    ax.add_patch(prot_slice)
    ax.text(11.55, 6.05, "5 PROTECTED LAYERS (FP16 or INT8)", fontsize=9.0, fontweight="bold", ha="center", va="center", color="#5B21B6")
    ax.text(11.55, 5.75, "390.6M Parameters | FP16 (16-bit) or INT8 (8-bit)\nKeeps critical safety decision boundaries intact",
            fontsize=7.5, ha="center", va="center", color="#4C1D95")

    # Bottom Slice: 31 Quantized Layers
    quant_slice = FancyBboxPatch((9.9, 4.6), 3.3, 0.75, boxstyle="round,pad=0.06,rounding_size=0.08",
                                 ec="#0D9488", fc="#CCFBF1", linewidth=1.2)
    ax.add_patch(quant_slice)
    ax.text(11.55, 5.08, "31 QUANTIZED LAYERS (NF4)", fontsize=9.0, fontweight="bold", ha="center", va="center", color="#0F766E")
    ax.text(11.55, 4.80, "2,421.7M Parameters | NormalFloat 4 (NF4)\nAchieves >60% layer compression efficiency",
            fontsize=7.5, ha="center", va="center", color="#115E59")

    # Quant Specs cards
    q_specs = [
        ("FP16 + NF4 Mode (_FP16_W4)", "Allocated: 3.12 GB VRAM (vs 6.70 GB FP16)\nThroughput: 128.9 tok/s | 53% Memory Reduction", "#DBEAFE", "#1E40AF"),
        ("INT8 + NF4 Mode (_W8_W4)", "Allocated: 2.73 GB VRAM (vs 2.53 GB Uniform NF4)\nThroughput: 111.5 tok/s | 90% Safety Recovery", "#EDE9FE", "#6D28D9"),
        ("Uniform NF4 Baseline", "Allocated: 2.53 GB VRAM (Degraded Refusal: 60.5%)\nSuffers 5.0 pp safety drop vs FP16 baseline", "#FEE2E2", "#991B1B"),
    ]
    for i, (q_t, q_d, q_bg, q_fg) in enumerate(q_specs):
        qy = 3.3 - i * 1.15
        q_card = FancyBboxPatch((9.7, qy), 3.7, 1.0, boxstyle="round,pad=0.06,rounding_size=0.1",
                                ec="#E2E8F0", fc="#FFFFFF", linewidth=1.0)
        ax.add_patch(q_card)
        q_tag = FancyBboxPatch((9.85, qy + 0.62), 3.4, 0.3, boxstyle="round,pad=0.03,rounding_size=0.06",
                               ec="none", fc=q_bg)
        ax.add_patch(q_tag)
        ax.text(11.55, qy + 0.77, q_t, fontsize=8.5, fontweight="bold", ha="center", va="center", color=q_fg)
        ax.text(11.55, qy + 0.32, q_d, fontsize=7.8, ha="center", va="center", color="#475569")

    # -------------------------------------------------------------
    # STAGE 4: EVALUATION & VALIDATION CONTENT
    # -------------------------------------------------------------
    eval_head = FancyBboxPatch((14.2, 7.0), 3.7, 0.75, boxstyle="round,pad=0.08,rounding_size=0.12",
                               ec="#FDE68A", fc="#FFFBEB", linewidth=1.2)
    ax.add_patch(eval_head)
    ax.text(16.05, 7.48, "Dual-Scorer Evaluation Protocol", fontsize=11, fontweight="bold", ha="center", va="center", color="#B45309")
    ax.text(16.05, 7.20, "Safety vs Capability Pareto Frontier Verification", fontsize=8.5, ha="center", va="center", color="#D97706")

    # 4 Validation Panels
    v_panels = [
        ("1. Safety Refusal (400 Prompts)",
         "Harmful Prompts (100 EN + 100 HI):\n- S_CK_circuitkit (FP16): 66.0% (+5.5 pp)\n- S_C_refusalDrop (FP16): 65.5% (+5.0 pp)\n- Uniform NF4: 60.5% (FP16 Baseline: 65.5%)",
         "#FEF3C7", "#92400E"),
        ("2. Capability Benchmarks",
         "Multilingual MMLU (600 Questions):\n- S_CK_circuitkit: 47.7% (Outperforms FP16 47.5%!)\nBelebele Comprehension (200 Questions):\n- FP16: 77.0% | S_C: 76.0% | Uniform NF4: 76.0%",
         "#E0F2FE", "#075985"),
        ("3. Independent Lexsi AuditKIT Scorer",
         "Independent Multilingual Evaluator (200 Prompts):\n- S_C_refusalDrop_FP16: 56.0% Refusal\n- Uniform NF4: 44.0% (+12.0 pp recovery!)\n95% Bootstrap CI: [+8.0 pp, +17.0 pp]",
         "#CCFBF1", "#0F766E"),
        ("4. Control & Ablation Verification",
         "Negative Control S_N: 64.0% (+3.5 pp).\nSize ablations (C3: 64.0%, C4: 63.0%)\nconfirm K=5 provides optimal boundary preservation.",
         "#FFE4E6", "#9F1239")
    ]

    for i, (v_t, v_d, v_bg, v_fg) in enumerate(v_panels):
        vy = 5.35 - i * 1.45
        v_card = FancyBboxPatch((14.2, vy), 3.7, 1.35, boxstyle="round,pad=0.06,rounding_size=0.1",
                                ec="#CBD5E1", fc="#FFFFFF", linewidth=1.0)
        ax.add_patch(v_card)
        v_tag = FancyBboxPatch((14.35, vy + 0.95), 3.4, 0.32, boxstyle="round,pad=0.03,rounding_size=0.06",
                               ec="none", fc=v_bg)
        ax.add_patch(v_tag)
        ax.text(16.05, vy + 1.11, v_t, fontsize=8.8, fontweight="bold", ha="center", va="center", color=v_fg)
        ax.text(16.05, vy + 0.46, v_d, fontsize=7.5, ha="center", va="center", color="#334155")

    # -------------------------------------------------------------
    # INTER-STAGE CONNECTING ARROWS
    # -------------------------------------------------------------
    arrow_style = ArrowStyle("Simple,head_width=5,head_length=8")
    arrows = [
        (4.65, 5.0, 4.95, 5.0),
        (9.15, 5.0, 9.45, 5.0),
        (13.65, 5.0, 13.95, 5.0)
    ]
    for x1, y1, x2, y2 in arrows:
        arr = patches.FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=arrow_style,
                                      color="#94A3B8", linewidth=2.0)
        ax.add_patch(arr)

    # Footer Notes
    ft_box = FancyBboxPatch((0.5, 0.2), 18.0, 0.45, boxstyle="round,pad=0.05,rounding_size=0.08",
                            ec="#E2E8F0", fc="#F8FAFC", linewidth=1.0)
    ax.add_patch(ft_box)
    ax.text(9.5, 0.42, "SafeCompress Research Pipeline • Lexsi Hackathon (Track 2b: AI Safety and Correction) • Author: Ankit Kumar Tiwari • Base Model: CohereLabs/tiny-aya-global",
            fontsize=9.0, ha="center", va="center", color="#64748B", fontfamily="sans-serif")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, facecolor="#FFFFFF", edgecolor="none", bbox_inches="tight")
    plt.close()
    print(f"Architecture diagram successfully saved to: {output_path}")

if __name__ == "__main__":
    create_architecture_diagram()
