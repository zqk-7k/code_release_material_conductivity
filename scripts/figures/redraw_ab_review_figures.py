from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


PAPER = Path(__file__).resolve().parents[1]
IMG = PAPER / "images"


def save(fig, stem):
    fig.savefig(IMG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(IMG / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def fig1():
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")
    boxes = [
        (0.04, 0.58, "Curated zirconia\nconductivity database\n1,351 records"),
        (0.27, 0.58, "Locked evaluation\nGrouped confirmation\nTemperature extrapolation"),
        (0.50, 0.58, "Arrhenius-constrained\nPIML\nEa and log10 A"),
        (0.73, 0.58, "Candidate prioritization\nSc–Mg follow-up region"),
        (0.73, 0.18, "Limited checks\nCHGNet MD\nDFT ΔEmix"),
    ]
    for x, y, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), 0.20, 0.24, fc="#F2F6FB", ec="#4C78A8", lw=1.5))
        ax.text(x + 0.10, y + 0.12, text, ha="center", va="center", fontsize=10)
    arrows = [((0.24, 0.70), (0.27, 0.70)), ((0.47, 0.70), (0.50, 0.70)), ((0.70, 0.70), (0.73, 0.70)), ((0.83, 0.58), (0.83, 0.42))]
    for xy, xytext in arrows:
        ax.annotate("", xy=xytext, xytext=xy, arrowprops=dict(arrowstyle="->", lw=1.5, color="#333333"))
    ax.text(0.50, 0.08, "Prioritization workflow with diagnostic computational checks", ha="center", fontsize=9, color="#555555")
    save(fig, "Fig1_database_evaluation_workflow_ab")


def fig3():
    left = Image.open(IMG / "paper_feature_importance_Ea.png").convert("RGB")
    right = Image.open(IMG / "Ea_vs_structure_and_doping.png").convert("RGB")
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), gridspec_kw={"width_ratios": [1, 1.3]})
    for ax in axes:
        ax.axis("off")
    axes[0].imshow(left)
    axes[1].imshow(right)
    axes[0].text(0.01, 0.98, "(a) permutation importance", transform=axes[0].transAxes, va="top", ha="left", fontsize=11, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))
    axes[1].text(0.02, 0.98, "(b) Ea vs average dopant radius", transform=axes[1].transAxes, va="top", ha="left", fontsize=10, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))
    axes[1].text(0.52, 0.98, "(c) Ea vs total dopant concentration", transform=axes[1].transAxes, va="top", ha="left", fontsize=10, weight="bold", bbox=dict(fc="white", ec="none", alpha=0.8))
    plt.tight_layout()
    save(fig, "Fig3_ea_interpretability_ab")


def fig5():
    labels = ["PIML-prioritized\nSc–Mg", "Ranked Y–Gd\ncomparator", "Mg-only\ncontrol", "Undoped\nZrO2"]
    piml = np.array([-1.04, -1.35, -2.10, -3.50])
    chgnet = np.array([-1.01, -1.17, -1.40, -3.92])
    err = np.array([0.12, 0.22, 0.16, 1.83])
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#9E9E9E"]
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    for x, y, e, c, lab in zip(piml, chgnet, err, colors, labels):
        ax.errorbar(x, y, yerr=e, fmt="o", ms=7, color=c, ecolor=c, capsize=3)
        ax.annotate(lab, (x, y), xytext=(5, 5), textcoords="offset points", fontsize=8)
    lim = (-6.4, -0.6)
    ax.plot(lim, lim, "--", color="black", lw=1, label="y = x")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("PIML predicted log10 σ (S/cm)")
    ax.set_ylabel("CHGNet MD log10 σ (S/cm)")
    ax.set_title("Ranking-level CHGNet comparison")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.25)
    ax.text(-6.25, -5.35, "Undoped ZrO2 shown with\nslow-diffusion floor handling", fontsize=8, color="#555555")
    save(fig, "Fig5_chgnet_ranking_comparison_ab")


def main():
    fig1()
    fig3()
    fig5()


if __name__ == "__main__":
    main()
