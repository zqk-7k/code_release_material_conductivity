from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper_write" / "conductivity"
IMG = PAPER / "images"
DATA = ROOT / "windows_material_conductivity_training_handoff" / "training_project" / "material-conductivity-data-clean_reference" / "data"
PRED = (
    ROOT
    / "windows_material_conductivity_training_handoff"
    / "windows_experiments"
    / "piml_metric_optimization"
    / "results"
    / "paper_compatible_multiseed"
    / "checkpoints"
    / "relu_bounded_seed_0"
    / "evaluation_predictions.csv"
)
OUT_DATA = PAPER / "results" / "ea_diagnostics_figure_data.csv"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10,
            "axes.labelsize": 10.5,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_figure_data() -> pd.DataFrame:
    pred = pd.read_csv(PRED)
    dop = pd.read_csv(DATA / "sample_dopants.tsv", sep="\t")
    radius = (
        dop.assign(weight=lambda d: d["dopant_molar_fraction"].fillna(0))
        .groupby("sample_id")
        .apply(
            lambda g: pd.Series(
                {
                    "average_dopant_radius": np.average(g["dopant_ionic_radius"], weights=np.clip(g["weight"], 1e-12, None)),
                    "total_dopant_fraction": g["dopant_molar_fraction"].sum(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    data = pred.merge(radius, on="sample_id", how="left")
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(OUT_DATA, index=False)
    return data


def main() -> None:
    style()
    df = load_figure_data()
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(8.8, 3.15),
        gridspec_kw={"width_ratios": [0.92, 1.05, 1.05]},
        constrained_layout=True,
    )

    # Panel (a) is rank-only: the archived source figure did not store numeric
    # permutation scores, so we preserve only the audited ranking used in the text.
    drivers = [
        "Sintering duration",
        "Number of dopants",
        "Max sintering temp.",
        "Avg dopant radius",
    ]
    ranks = np.arange(1, len(drivers) + 1)
    ax = axes[0]
    ax.scatter(ranks, drivers, s=42, color="#4C78A8", zorder=3)
    for rank, label in zip(ranks, drivers):
        ax.hlines(label, 1, rank, color="#C8D7EA", lw=2, zorder=1)
    ax.invert_yaxis()
    ax.set_xlim(0.7, 4.3)
    ax.set_xlabel("Permutation rank")
    ax.set_ylabel("")
    ax.set_xticks(ranks)
    ax.grid(axis="x", alpha=0.2)
    ax.text(-0.18, 1.08, "(a)", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[1]
    ax.scatter(df["average_dopant_radius"], df["predicted_Ea"], s=20, alpha=0.72, color="#59A14F", edgecolor="white", linewidth=0.25)
    ax.set_xlabel("Average dopant radius (pm)")
    ax.set_ylabel(r"Inferred $E_a$ (eV)")
    ax.grid(alpha=0.18)
    ax.text(-0.18, 1.08, "(b)", transform=ax.transAxes, fontsize=13, fontweight="bold")

    ax = axes[2]
    ax.scatter(100 * df["total_dopant_fraction"], df["predicted_Ea"], s=20, alpha=0.72, color="#F28E2B", edgecolor="white", linewidth=0.25)
    ax.set_xlabel("Total dopant fraction (mol%)")
    ax.set_ylabel(r"Inferred $E_a$ (eV)")
    ax.grid(alpha=0.18)
    ax.text(-0.18, 1.08, "(c)", transform=ax.transAxes, fontsize=13, fontweight="bold")

    IMG.mkdir(parents=True, exist_ok=True)
    fig.savefig(IMG / "Fig3_ea_diagnostics_jmateriomics.pdf", bbox_inches="tight")
    fig.savefig(IMG / "Fig3_ea_diagnostics_jmateriomics.png", dpi=600, bbox_inches="tight")
    print("Wrote", IMG / "Fig3_ea_diagnostics_jmateriomics.pdf")
    print("Wrote", OUT_DATA)


if __name__ == "__main__":
    main()
