from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper_write" / "conductivity"
IMG = PAPER / "images"
SURFACE = ROOT / "windows_material_conductivity_training_handoff" / "existing_results" / "paper_ready_inverse_design" / "results" / "04_surface_sc_mg_T1505.csv"
IMG.mkdir(parents=True, exist_ok=True)


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig, stem: str) -> None:
    fig.savefig(IMG / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(IMG / f"{stem}.png", dpi=600, bbox_inches="tight")
    plt.close(fig)


def workflow() -> None:
    fig, ax = plt.subplots(figsize=(7.8, 4.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(x, y, w, h, title, body, fc, ec):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            fc=fc,
            ec=ec,
            lw=1.15,
        )
        ax.add_patch(patch)
        ax.text(x + 0.018, y + h - 0.045, title, ha="left", va="top", fontsize=9.5, weight="bold", color="#222222")
        ax.text(x + 0.018, y + h - 0.095, body, ha="left", va="top", fontsize=8.4, color="#333333", linespacing=1.25)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="-|>", lw=1.25, color="#404040", shrinkA=3, shrinkB=3))

    box(
        0.03,
        0.62,
        0.19,
        0.26,
        "Data sources",
        "In-house EIS\nLiterature tables\nDOI provenance",
        "#EAF2FB",
        "#4C78A8",
    )
    box(
        0.28,
        0.62,
        0.20,
        0.26,
        "Curation",
        "String-preserving buffer\nRule-based numerical extraction\nText normalization",
        "#F5F7FA",
        "#6B7280",
    )
    box(
        0.54,
        0.62,
        0.18,
        0.26,
        "Relational database",
        "Sample table\nDopant records\nSintering steps",
        "#EAF7EF",
        "#54A24B",
    )
    box(
        0.78,
        0.62,
        0.18,
        0.26,
        "Feature matrix",
        "Composition\nProcessing\nText-context metadata",
        "#FFF4E6",
        "#D98C00",
    )
    box(
        0.18,
        0.18,
        0.23,
        0.24,
        "Locked evaluation",
        "Grouped confirmation\nHistorical benchmark\nTemperature extrapolation",
        "#F4F0FA",
        "#7B61A8",
    )
    box(
        0.47,
        0.18,
        0.22,
        0.24,
        "Arrhenius learning",
        "PIML vs DNN\nBounded effective $E_a$\n1000/T diagnostic",
        "#EEF6FF",
        "#2F6FA3",
    )
    box(
        0.75,
        0.18,
        0.20,
        0.24,
        "Hypothesis checks",
        "Fixed-run GA region\nEmpirical screen\nCHGNet/DFT ranking",
        "#FDF1F1",
        "#C84B4B",
    )

    arrow((0.22, 0.75), (0.28, 0.75))
    arrow((0.48, 0.75), (0.54, 0.75))
    arrow((0.72, 0.75), (0.78, 0.75))
    arrow((0.87, 0.62), (0.87, 0.44))
    arrow((0.78, 0.31), (0.69, 0.31))
    arrow((0.47, 0.31), (0.41, 0.31))
    ax.text(0.03, 0.05, "Workflow output: curated records, locked benchmarks, temperature-extrapolation diagnostics and Sc-rich/Mg-containing follow-up hypotheses.", fontsize=8.6, color="#333333")
    save(fig, "Fig1_jmateriomics_workflow")


def architecture() -> None:
    fig, ax = plt.subplots(figsize=(7.6, 3.4))
    ax.set_axis_off()
    colors = {"input": "#E8F1FA", "net": "#F4F4F4", "phys": "#FFF1D6", "out": "#EAF5EA"}
    boxes = [
        (0.04, 0.56, 0.18, 0.25, "Composition,\nprocessing,\ntext context", colors["input"]),
        (0.30, 0.56, 0.18, 0.25, "MLP encoder\n256-128-64", colors["net"]),
        (0.55, 0.68, 0.14, 0.16, "$E_a$\n0.03--2.00 eV", colors["out"]),
        (0.56, 0.46, 0.13, 0.16, r"$\log_{10} A$", colors["out"]),
        (0.73, 0.52, 0.25, 0.30, "Arrhenius layer\n" + r"$\log_{10}\sigma=\log_{10}A-\log_{10}T$" + "\n" + r"$-E_a/(k_BT\ln 10)$", colors["phys"]),
        (0.30, 0.18, 0.18, 0.17, "Temperature\n$T$ only here", colors["input"]),
    ]
    for x, y, w, h, label, fc in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="#555555", lw=1.0))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9)
    arrows = [
        ((0.22, 0.68), (0.30, 0.68)),
        ((0.48, 0.68), (0.55, 0.76)),
        ((0.48, 0.68), (0.56, 0.54)),
        ((0.69, 0.76), (0.73, 0.67)),
        ((0.69, 0.54), (0.73, 0.61)),
        ((0.48, 0.27), (0.73, 0.55)),
    ]
    for a, b in arrows:
        ax.annotate("", xy=b, xytext=a, arrowprops=dict(arrowstyle="->", lw=1.1, color="#333333"))
    ax.text(0.04, 0.08, "Encoder excludes measurement temperature; material context determines $E_a$ and $\\log_{10}A$.", fontsize=9)
    save(fig, "Fig2_jmateriomics_piml_architecture")


def ga_surface() -> None:
    milestones = pd.DataFrame(
        {
            "generation": [0, 5, 24],
            "best_score": [-1.228, -1.058, -1.036],
            "source": [
                "manuscript/supplementary fixed GA run",
                "manuscript/supplementary fixed GA run",
                "manuscript/supplementary fixed GA run",
            ],
        }
    )
    out = PAPER / "results" / "jmateriomics_fig4_fixed_run_milestones.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    milestones.to_csv(out, index=False)
    surf = pd.read_csv(SURFACE)
    surf = surf[(surf["Sc_mol_percent"].between(5, 12)) & (surf["Mg_mol_percent"].between(2.8, 7.2))]
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), constrained_layout=True)
    ax = axes[0]
    ax.plot(milestones["generation"], milestones["best_score"], color="#1F4E79", lw=1.7, marker="o", ms=4)
    for _, row in milestones.iterrows():
        ax.annotate(f"{row['best_score']:.3f}", (row["generation"], row["best_score"]), xytext=(3, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("GA generation")
    ax.set_ylabel(r"Best predicted $\log_{10}\sigma$")
    ax.grid(alpha=0.25)
    ax.text(-0.12, 1.08, "(a)", transform=ax.transAxes, fontweight="bold", fontsize=12)

    ax = axes[1]
    piv = surf.pivot_table(index="Mg_mol_percent", columns="Sc_mol_percent", values="predicted_log10_sigma")
    im = ax.imshow(
        piv.values,
        origin="lower",
        aspect="auto",
        extent=[piv.columns.min(), piv.columns.max(), piv.index.min(), piv.index.max()],
        cmap="viridis",
    )
    ax.scatter([7.50], [3.19], marker="*", s=90, color="#D62728", edgecolor="white", linewidth=0.6)
    ax.annotate("representative\ncandidate", (7.50, 3.19), xytext=(6, 8), textcoords="offset points", fontsize=8, color="#222222")
    ax.set_xlabel("Sc concentration (mol%)")
    ax.set_ylabel("Mg concentration (mol%)")
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.02)
    cb.set_label(r"Predicted $\log_{10}\sigma$")
    ax.text(-0.12, 1.08, "(b)", transform=ax.transAxes, fontweight="bold", fontsize=12)
    save(fig, "Fig4_jmateriomics_candidate_hypothesis")


def chgnet() -> None:
    labels = ["Sc-Mg\nregion", "Y-Gd\ncomparator", "Mg-only\ncontrol", "Undoped\nZrO$_2$"]
    piml = np.array([-1.04, -1.35, -2.10, -3.50])
    md = np.array([-1.01, -1.17, -1.40, -3.92])
    err = np.array([0.12, 0.22, 0.16, 1.83])
    colors = ["#4C78A8", "#72B7B2", "#F58518", "#9E9E9E"]
    offsets_full = {
        "Sc-Mg\nregion": (18, -34),
        "Y-Gd\ncomparator": (-88, -42),
        "Mg-only\ncontrol": (-52, -28),
        "Undoped\nZrO$_2$": (12, -24),
    }
    offsets_zoom = {
        "Sc-Mg\nregion": (18, -30),
        "Y-Gd\ncomparator": (-70, 18),
        "Mg-only\ncontrol": (12, 18),
        "Undoped\nZrO$_2$": (10, -18),
    }
    fig, axes = plt.subplots(1, 2, figsize=(7.8, 3.6), gridspec_kw={"width_ratios": [1.05, 1.0]}, constrained_layout=True)
    for ax, lim, title, offsets in [
        (axes[0], (-4.8, -0.6), "Full range", offsets_full),
        (axes[1], (-2.35, -0.75), "Doped-composition zoom", offsets_zoom),
    ]:
        for x, y, e, c, lab in zip(piml, md, err, colors, labels):
            ax.errorbar(x, y, yerr=e, fmt="o", ms=5.5, color=c, ecolor=c, capsize=2.5)
            if lim[0] <= x <= lim[1] and lim[0] <= y <= lim[1]:
                if title == "Full range" and lab == "Y-Gd\ncomparator":
                    continue
                ax.annotate(
                    lab,
                    (x, y),
                    xytext=offsets[lab],
                    textcoords="offset points",
                    fontsize=8.3,
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.82),
                    arrowprops=dict(arrowstyle="-", lw=0.5, color="#666666", shrinkA=2, shrinkB=2),
                )
        ax.plot(lim, lim, "--", color="#333333", lw=0.9)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(r"PIML $\log_{10}\sigma$")
        ax.grid(alpha=0.22)
    axes[0].set_ylabel(r"CHGNet MD $\log_{10}\sigma$")
    axes[0].text(-0.14, 1.08, "(a)", transform=axes[0].transAxes, fontweight="bold", fontsize=12)
    axes[1].text(-0.14, 1.08, "(b)", transform=axes[1].transAxes, fontweight="bold", fontsize=12)
    save(fig, "Fig5_jmateriomics_chgnet_comparison")


def main() -> None:
    style()
    workflow()
    architecture()
    ga_surface()
    chgnet()
    print("Wrote Journal of Materiomics figures to", IMG)


if __name__ == "__main__":
    main()
