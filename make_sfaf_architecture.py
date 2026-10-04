from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


# ============================================================
# SFAF ARCHITECTURE FIGURE
# Journal-ready version
#
# Outputs:
#   figures/sfaf_architecture.png
#   figures/sfaf_architecture.svg
# ============================================================

PROJECT_DIR = Path(r"E:\Papers\AgriVision Blueberry")
OUT_DIR = PROJECT_DIR / "figures"

PNG_PATH = OUT_DIR / "sfaf_architecture.png"
SVG_PATH = OUT_DIR / "sfaf_architecture.svg"

DPI = 600


# ============================================================
# GLOBAL FONT SETTINGS
# ============================================================

plt.rcParams["font.family"] = "Times New Roman"

# Use Times New Roman also for mathematical text
plt.rcParams["mathtext.fontset"] = "custom"
plt.rcParams["mathtext.rm"] = "Times New Roman"
plt.rcParams["mathtext.it"] = "Times New Roman:italic"
plt.rcParams["mathtext.bf"] = "Times New Roman:bold"


# ============================================================
# FIGURE HELPERS
# ============================================================

def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    facecolor="white",
    edgecolor="0.20",
    linewidth=1.25,
    fontsize=10.5,
    radius=0.08,
):
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=(
            f"round,pad=0.015,"
            f"rounding_size={radius}"
        ),
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=linewidth,
        zorder=2,
    )

    ax.add_patch(patch)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        fontfamily="Times New Roman",
        color="black",
        linespacing=1.10,
        zorder=3,
    )

    return patch


def add_arrow(
    ax,
    start,
    end,
    linewidth=1.15,
    color="0.20",
    connectionstyle="arc3,rad=0",
    mutation_scale=11,
):
    arrow = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=linewidth,
        color=color,
        connectionstyle=connectionstyle,
        shrinkA=1,
        shrinkB=1,
        zorder=1,
    )

    ax.add_patch(arrow)


def add_line(
    ax,
    x1,
    y1,
    x2,
    y2,
    linewidth=1.15,
    color="0.20",
):
    ax.plot(
        [x1, x2],
        [y1, y2],
        linewidth=linewidth,
        color=color,
        zorder=1,
    )


def add_orthogonal_arrow(
    ax,
    points,
    linewidth=1.15,
    color="0.20",
):
    """
    Draw a multi-segment path with an arrowhead
    on the final segment.
    """

    for i in range(len(points) - 2):

        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        add_line(
            ax,
            x1,
            y1,
            x2,
            y2,
            linewidth=linewidth,
            color=color,
        )

    add_arrow(
        ax,
        points[-2],
        points[-1],
        linewidth=linewidth,
        color=color,
    )


# ============================================================
# MAIN FIGURE
# ============================================================

def make_figure():

    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Wide layout suitable for manuscript insertion
    fig, ax = plt.subplots(
        figsize=(16.5, 7.6)
    )

    ax.set_xlim(0, 31)
    ax.set_ylim(0, 12.7)

    ax.axis("off")


    # ========================================================
    # COLORS
    # ========================================================

    c_input = "#EAF2F8"
    c_encoder = "#DCEAF7"

    c_feature = "#F2E6F4"
    c_projection = "#FFF2CC"

    c_branch = "#DDEFD8"

    c_gate = "#F7DDDF"
    c_fusion = "#F9E4AD"

    c_refine = "#E4F0DC"
    c_output = "#DCEBF7"

    c_neutral = "#F4F4F4"


    # ========================================================
    # INPUT
    # ========================================================

    add_box(
        ax,
        0.30,
        8.00,
        2.15,
        1.15,
        (
            "Input image\n"
            "640 × 640 × 3"
        ),
        facecolor=c_input,
        fontsize=11,
    )


    # ========================================================
    # MiT-B0 ENCODER
    # ========================================================

    add_box(
        ax,
        2.95,
        7.25,
        3.10,
        2.65,
        (
            "MiT-B0 encoder\n"
            "ImageNet-pretrained\n"
            "unchanged"
        ),
        facecolor=c_encoder,
        fontsize=11.5,
    )

    add_arrow(
        ax,
        (2.45, 8.57),
        (2.95, 8.57),
    )


    # ========================================================
    # ENCODER FEATURE MAPS
    # ========================================================

    feature_x = 6.85
    feature_w = 2.95
    feature_h = 1.15

    feature_y = {
        "C1": 10.00,
        "C2": 7.90,
        "C3": 5.80,
        "C4": 3.70,
    }

    feature_text = {

        "C1": (
            "C1\n"
            "160 × 160 × 32\n"
            "stage 1 (H/4)"
        ),

        "C2": (
            "C2\n"
            "80 × 80 × 64\n"
            "stage 2 (H/8)"
        ),

        "C3": (
            "C3\n"
            "40 × 40 × 160\n"
            "stage 3 (H/16)"
        ),

        "C4": (
            "C4\n"
            "20 × 20 × 256\n"
            "stage 4 (H/32)"
        ),
    }


    for key in ["C1", "C2", "C3", "C4"]:

        y = feature_y[key]

        add_box(
            ax,
            feature_x,
            y,
            feature_w,
            feature_h,
            feature_text[key],
            facecolor=c_feature,
            fontsize=10.1,
        )

        add_arrow(
            ax,
            (6.05, 8.57),
            (
                feature_x,
                y + feature_h / 2,
            ),
        )


    # ========================================================
    # 1 × 1 PROJECTIONS
    # ========================================================

    projection_x = 10.55
    projection_w = 2.55

    projection_text = {

        "C1": (
            "1 × 1 projection\n"
            "P1: 64 ch"
        ),

        "C2": (
            "1 × 1 projection\n"
            "P2: 64 ch"
        ),

        "C3": (
            "1 × 1 projection\n"
            "P3: 64 ch"
        ),

        "C4": (
            "1 × 1 projection\n"
            "P4: 64 ch"
        ),
    }


    for key in ["C1", "C2", "C3", "C4"]:

        y = feature_y[key]

        add_box(
            ax,
            projection_x,
            y,
            projection_w,
            feature_h,
            projection_text[key],
            facecolor=c_projection,
            fontsize=10.1,
        )

        add_arrow(
            ax,
            (
                feature_x + feature_w,
                y + feature_h / 2,
            ),
            (
                projection_x,
                y + feature_h / 2,
            ),
        )


    # ========================================================
    # RESOLUTION ALIGNMENT
    # ========================================================

    resize_x = 13.80
    resize_w = 2.35

    # P1 is already 160 × 160 and therefore
    # requires no resize operation.

    for key in ["C2", "C3", "C4"]:

        y = feature_y[key]

        add_box(
            ax,
            resize_x,
            y,
            resize_w,
            feature_h,
            (
                "Bilinear resize\n"
                "to 160 × 160"
            ),
            facecolor=c_neutral,
            fontsize=9.8,
        )

        add_arrow(
            ax,
            (
                projection_x + projection_w,
                y + feature_h / 2,
            ),
            (
                resize_x,
                y + feature_h / 2,
            ),
        )


    # ========================================================
    # DETAIL STREAM
    # ========================================================

    branch_x = 17.25
    branch_w = 3.60
    branch_h = 1.65

    detail_y = 8.65

    add_box(
        ax,
        branch_x,
        detail_y,
        branch_w,
        branch_h,
        (
            "Detail stream D\n"
            "Concat(P1, P2↑)\n"
            "Depthwise-separable conv"
        ),
        facecolor=c_branch,
        fontsize=10.5,
    )


    # P1 -> detail stream
    add_arrow(
        ax,
        (
            projection_x + projection_w,
            feature_y["C1"] + feature_h / 2,
        ),
        (
            branch_x,
            detail_y + 1.12,
        ),
    )


    # P2 -> detail stream
    add_arrow(
        ax,
        (
            resize_x + resize_w,
            feature_y["C2"] + feature_h / 2,
        ),
        (
            branch_x,
            detail_y + 0.50,
        ),
    )


    # ========================================================
    # SEMANTIC STREAM
    # ========================================================

    semantic_y = 4.85

    add_box(
        ax,
        branch_x,
        semantic_y,
        branch_w,
        branch_h,
        (
            "Semantic stream S\n"
            "Concat(P3↑, P4↑)\n"
            "Depthwise-separable conv"
        ),
        facecolor=c_branch,
        fontsize=10.5,
    )


    # P3 -> semantic stream
    add_arrow(
        ax,
        (
            resize_x + resize_w,
            feature_y["C3"] + feature_h / 2,
        ),
        (
            branch_x,
            semantic_y + 1.12,
        ),
    )


    # P4 -> semantic stream
    add_arrow(
        ax,
        (
            resize_x + resize_w,
            feature_y["C4"] + feature_h / 2,
        ),
        (
            branch_x,
            semantic_y + 0.50,
        ),
    )


    # ========================================================
    # GATE GENERATOR
    # ========================================================

    gate_x = 22.05
    gate_y = 6.90

    gate_w = 3.45
    gate_h = 2.35

    add_box(
        ax,
        gate_x,
        gate_y,
        gate_w,
        gate_h,
        (
            "Gate generator\n"
            "Concat(D, S)\n"
            "1 × 1 Conv → BN → GELU\n"
            "1 × 1 Conv → Sigmoid\n"
            "64-channel spatial gate G"
        ),
        facecolor=c_gate,
        fontsize=9.8,
    )


    # Detail stream -> gate
    add_arrow(
        ax,
        (
            branch_x + branch_w,
            detail_y + branch_h / 2,
        ),
        (
            gate_x,
            gate_y + 1.72,
        ),
    )


    # Semantic stream -> gate
    add_arrow(
        ax,
        (
            branch_x + branch_w,
            semantic_y + branch_h / 2,
        ),
        (
            gate_x,
            gate_y + 0.62,
        ),
    )


    # ========================================================
    # ADAPTIVE FUSION
    # ========================================================

    fusion_x = 26.65
    fusion_y = 7.15

    fusion_w = 4.05
    fusion_h = 1.60

    add_box(
        ax,
        fusion_x,
        fusion_y,
        fusion_w,
        fusion_h,
        (
            "Adaptive fusion\n"
            r"$F = G \odot D + (1-G)\odot S$"
        ),
        facecolor=c_fusion,
        fontsize=11,
    )


    # Gate -> adaptive fusion
    add_arrow(
        ax,
        (
            gate_x + gate_w,
            gate_y + gate_h / 2,
        ),
        (
            fusion_x,
            fusion_y + fusion_h / 2,
        ),
    )


    # Direct detail stream -> fusion
    add_arrow(
        ax,
        (
            branch_x + branch_w,
            detail_y + 1.02,
        ),
        (
            fusion_x,
            fusion_y + 1.27,
        ),
        connectionstyle="arc3,rad=-0.08",
    )


    # Direct semantic stream -> fusion
    add_arrow(
        ax,
        (
            branch_x + branch_w,
            semantic_y + 0.62,
        ),
        (
            fusion_x,
            fusion_y + 0.30,
        ),
        connectionstyle="arc3,rad=0.08",
    )


    # ========================================================
    # P1 REINJECTION
    # ========================================================

    reinject_x = 26.85
    reinject_y = 9.85

    reinject_w = 3.70
    reinject_h = 1.15

    add_box(
        ax,
        reinject_x,
        reinject_y,
        reinject_w,
        reinject_h,
        "Concatenate [F, P1]",
        facecolor=c_projection,
        fontsize=10.8,
    )


    # Adaptive fusion -> concatenation
    add_arrow(
        ax,
        (
            fusion_x + fusion_w / 2,
            fusion_y + fusion_h,
        ),
        (
            reinject_x + reinject_w / 2,
            reinject_y,
        ),
    )


    # --------------------------------------------------------
    # P1 skip connection
    # --------------------------------------------------------

    p1_source = (
        projection_x + projection_w,
        feature_y["C1"] + feature_h / 2,
    )

    add_orthogonal_arrow(
        ax,
        [
            p1_source,
            (
                13.40,
                p1_source[1],
            ),
            (
                13.40,
                11.75,
            ),
            (
                reinject_x + reinject_w / 2,
                11.75,
            ),
            (
                reinject_x + reinject_w / 2,
                reinject_y + reinject_h,
            ),
        ],
    )


    # ========================================================
    # REFINEMENT BLOCK
    # ========================================================

    output_x = 26.85
    output_w = 3.70

    refine_y = 5.05

    add_box(
        ax,
        output_x,
        refine_y,
        output_w,
        1.25,
        (
            "Refinement block\n"
            "Depthwise-separable conv"
        ),
        facecolor=c_refine,
        fontsize=10.4,
    )


    # Concatenation -> refinement
    add_arrow(
        ax,
        (
            reinject_x + reinject_w / 2,
            reinject_y,
        ),
        (
            output_x + output_w / 2,
            refine_y + 1.25,
        ),
    )


    # ========================================================
    # DROPOUT
    # ========================================================

    dropout_y = 3.55

    add_box(
        ax,
        output_x,
        dropout_y,
        output_w,
        1.00,
        "Dropout2d (p = 0.1)",
        facecolor=c_neutral,
        fontsize=10.5,
    )


    add_arrow(
        ax,
        (
            output_x + output_w / 2,
            refine_y,
        ),
        (
            output_x + output_w / 2,
            dropout_y + 1.00,
        ),
    )


    # ========================================================
    # CLASSIFIER
    # ========================================================

    classifier_y = 2.05

    add_box(
        ax,
        output_x,
        classifier_y,
        output_w,
        1.05,
        (
            "1 × 1 classifier\n"
            "2-class logits: 160 × 160"
        ),
        facecolor=c_output,
        fontsize=10.3,
    )


    add_arrow(
        ax,
        (
            output_x + output_w / 2,
            dropout_y,
        ),
        (
            output_x + output_w / 2,
            classifier_y + 1.05,
        ),
    )


    # ========================================================
    # BILINEAR OUTPUT RESIZE
    # ========================================================

    resize_output_y = 0.55

    add_box(
        ax,
        output_x,
        resize_output_y,
        output_w,
        1.05,
        (
            "Bilinear resize\n"
            "640 × 640"
        ),
        facecolor=c_output,
        fontsize=10.4,
    )


    add_arrow(
        ax,
        (
            output_x + output_w / 2,
            classifier_y,
        ),
        (
            output_x + output_w / 2,
            resize_output_y + 1.05,
        ),
    )


    # ========================================================
    # FINAL SEGMENTATION MASK
    # ========================================================

    final_x = 22.15
    final_y = 0.55

    final_w = 3.40
    final_h = 1.05

    add_box(
        ax,
        final_x,
        final_y,
        final_w,
        final_h,
        (
            "Argmax\n"
            "blueberry / background mask"
        ),
        facecolor=c_input,
        fontsize=10.5,
    )


    add_arrow(
        ax,
        (
            output_x,
            resize_output_y + 0.525,
        ),
        (
            final_x + final_w,
            final_y + final_h / 2,
        ),
    )


    # ========================================================
    # UPSAMPLING NOTATION
    # ========================================================

    ax.text(
        19.0,
        4.32,
        "↑ : bilinear resize to 160 × 160",
        ha="center",
        va="center",
        fontsize=9.5,
        fontfamily="Times New Roman",
        color="0.28",
    )


    # ========================================================
    # BOTTOM EXPLANATORY NOTE
    # ========================================================

    ax.text(
        15.5,
        0.03,
        (
            "All projected encoder features contain 64 channels. "
            "The decoder groups shallow features into a detail stream "
            "and deeper features into a semantic stream, followed by "
            "learned convex fusion and P1 feature reinjection."
        ),
        ha="center",
        va="bottom",
        fontsize=9.5,
        fontfamily="Times New Roman",
        color="0.22",
    )


    # ========================================================
    # SAVE
    # ========================================================

    plt.tight_layout(
        pad=0.3
    )

    fig.savefig(
        PNG_PATH,
        dpi=DPI,
        bbox_inches="tight",
        facecolor="white",
    )

    fig.savefig(
        SVG_PATH,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)


    print()
    print("SFAF architecture figure created successfully.")
    print(f"PNG: {PNG_PATH}")
    print(f"SVG: {SVG_PATH}")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    make_figure()