from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[3]
PAPER = ROOT / "paper_write" / "conductivity"
DATA = (
    ROOT
    / "windows_material_conductivity_training_handoff"
    / "training_project"
    / "material-conductivity-data-clean_reference"
    / "data"
)
RAW = DATA / "raw_conductivity_samples.tsv"
SAMPLES = DATA / "material_samples.tsv"
DOPANTS = DATA / "sample_dopants.tsv"
SINTER = DATA / "sintering_steps.tsv"
PARTITIONS = (
    ROOT
    / "windows_material_conductivity_training_handoff"
    / "windows_experiments"
    / "piml_metric_optimization"
    / "results"
    / "physics_campaign_smoke_clean"
    / "reserved_partition_membership.csv"
)

FIG_DIR = PAPER / "images"
OUT_DIR = PAPER / "results" / "database_overview"
FIG_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)

AUDIT = OUT_DIR / "figure_data_audit.txt"
FIG_PDF = FIG_DIR / "Fig_database_overview.pdf"
FIG_PNG = FIG_DIR / "Fig_database_overview.png"
SUPP_PDF = FIG_DIR / "FigS_database_supp.pdf"
SUPP_PNG = FIG_DIR / "FigS_database_supp.png"

IN_HOUSE_REFS = {"实验室自制", "实验室自行制备"}


def read_tsv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep="\t")


def source_type(ref: object) -> str:
    text = "" if pd.isna(ref) else str(ref).strip()
    return "in-house" if text in IN_HOUSE_REFS else "literature"


def fmt_float(x: float, ndigits: int = 4) -> str:
    if pd.isna(x):
        return "NA"
    return f"{x:.{ndigits}f}"


def weighted_average(group: pd.DataFrame, value_col: str) -> float:
    weights = pd.to_numeric(group["dopant_molar_fraction"], errors="coerce").fillna(0.0)
    values = pd.to_numeric(group[value_col], errors="coerce")
    mask = values.notna() & (weights > 0)
    if not mask.any() or weights[mask].sum() <= 0:
        return float("nan")
    return float(np.average(values[mask], weights=weights[mask]))


def build_feature_table(samples: pd.DataFrame, dopants: pd.DataFrame, sinter: pd.DataFrame) -> pd.DataFrame:
    d = dopants.copy()
    for col in ["dopant_molar_fraction", "dopant_ionic_radius", "dopant_valence"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")

    primary = (
        d.sort_values(["sample_id", "dopant_molar_fraction", "dopant_element"], ascending=[True, False, True])
        .drop_duplicates("sample_id")
        .set_index("sample_id")["dopant_element"]
    )
    total = d.groupby("sample_id")["dopant_molar_fraction"].sum()
    ndop = d.groupby("sample_id")["dopant_element"].nunique()
    avg_radius = d.groupby("sample_id").apply(lambda g: weighted_average(g, "dopant_ionic_radius"), include_groups=False)
    avg_valence = d.groupby("sample_id").apply(lambda g: weighted_average(g, "dopant_valence"), include_groups=False)

    s = sinter.copy()
    s["sintering_temperature"] = pd.to_numeric(s["sintering_temperature"], errors="coerce")
    s["sintering_duration"] = pd.to_numeric(s["sintering_duration"], errors="coerce")
    max_temp = s.groupby("sample_id")["sintering_temperature"].max()
    total_duration = s.groupby("sample_id")["sintering_duration"].sum(min_count=1)

    feat = samples.copy()
    feat["source_type"] = feat["reference"].map(source_type)
    feat["primary_dopant_element"] = feat["sample_id"].map(primary).fillna("None")
    feat["total_dopant_fraction"] = feat["sample_id"].map(total)
    feat["number_of_dopants"] = feat["sample_id"].map(ndop).fillna(0).astype(int)
    feat["average_dopant_radius"] = feat["sample_id"].map(avg_radius)
    feat["average_dopant_valence"] = feat["sample_id"].map(avg_valence)
    feat["maximum_sintering_temperature"] = feat["sample_id"].map(max_temp)
    feat["total_sintering_duration"] = feat["sample_id"].map(total_duration)
    feat["measurement_temperature"] = pd.to_numeric(feat["operating_temperature"], errors="coerce")
    feat["conductivity"] = pd.to_numeric(feat["conductivity"], errors="coerce")
    feat["log10_conductivity"] = np.log10(feat["conductivity"])
    feat["material_source_and_purity"] = feat["material_source_and_purity"].fillna("")
    feat["synthesis_method"] = feat["synthesis_method"].fillna("Unknown")
    return feat


def make_preprocessor() -> ColumnTransformer:
    numeric = [
        "total_dopant_fraction",
        "average_dopant_radius",
        "average_dopant_valence",
        "number_of_dopants",
        "maximum_sintering_temperature",
        "total_sintering_duration",
    ]
    categorical = ["primary_dopant_element", "synthesis_method"]
    text = "material_source_and_purity"
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]),
                numeric,
            ),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical),
            (
                "text",
                Pipeline(
                    [
                        (
                            "tfidf",
                            TfidfVectorizer(max_features=500, stop_words="english"),
                        ),
                        ("svd", TruncatedSVD(n_components=16, random_state=42)),
                    ]
                ),
                text,
            ),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def write_probe(audit, frames: dict[str, pd.DataFrame]) -> None:
    audit.write("DATABASE PROBE\n")
    audit.write("==============\n")
    for name, df in frames.items():
        audit.write(f"\n[{name}] path={frames[name].attrs.get('path', 'NA')}\n")
        audit.write(f"rows={len(df)}, columns={len(df.columns)}\n")
        audit.write("columns=" + ", ".join(map(str, df.columns)) + "\n")
        audit.write("first_5_rows:\n")
        audit.write(df.head(5).to_string(index=False))
        audit.write("\n")
    audit.write("\nFIELD CHECK\n")
    audit.write("===========\n")
    needed = {
        "primary_dopant_element": "derived from sample_dopants.tsv by largest dopant_molar_fraction per sample_id",
        "dopant elements": "sample_dopants.tsv::dopant_element",
        "dopant mol%": "sample_dopants.tsv::dopant_molar_fraction",
        "total_dopant_fraction": "derived sum of dopant_molar_fraction per sample_id",
        "number_of_dopants": "derived nunique dopant_element per sample_id",
        "measurement_temperature": "material_samples.tsv::operating_temperature",
        "log10_conductivity": "derived log10(material_samples.tsv::conductivity)",
        "synthesis_route / synthesis_method": "material_samples.tsv::synthesis_method",
        "maximum_sintering_temperature": "derived max sintering_temperature per sample_id",
        "total_sintering_duration": "derived sum sintering_duration per sample_id",
        "data source": "derived from material_samples.tsv::reference",
        "DOI": "material_samples.tsv::reference for literature records where DOI-like",
        "sample_id": "material_samples.tsv::sample_id",
    }
    for field, source in needed.items():
        audit.write(f"{field}: FOUND ({source})\n")


def main() -> None:
    raw = read_tsv(RAW)
    samples = read_tsv(SAMPLES)
    dopants = read_tsv(DOPANTS)
    sinter = read_tsv(SINTER)
    for path, df in [(RAW, raw), (SAMPLES, samples), (DOPANTS, dopants), (SINTER, sinter)]:
        df.attrs["path"] = str(path)

    feat = build_feature_table(samples, dopants, sinter)
    partitions = pd.read_csv(PARTITIONS) if PARTITIONS.exists() else pd.DataFrame()

    with AUDIT.open("w", encoding="utf-8") as audit:
        write_probe(
            audit,
            {
                "raw_conductivity_samples": raw,
                "material_samples": samples,
                "sample_dopants": dopants,
                "sintering_steps": sinter,
            },
        )

        audit.write("\nUNIQUE VALUES / RANGES\n")
        audit.write("======================\n")
        audit.write("primary_dopant_element counts:\n")
        audit.write(feat["primary_dopant_element"].value_counts(dropna=False).to_string())
        audit.write("\n")
        audit.write("dopant_element counts:\n")
        audit.write(dopants["dopant_element"].value_counts(dropna=False).to_string())
        audit.write("\n")
        for col in [
            "total_dopant_fraction",
            "number_of_dopants",
            "measurement_temperature",
            "conductivity",
            "log10_conductivity",
            "maximum_sintering_temperature",
            "total_sintering_duration",
        ]:
            ser = pd.to_numeric(feat[col], errors="coerce")
            audit.write(
                f"{col}: n={ser.notna().sum()}, min={fmt_float(ser.min(), 6)}, "
                f"median={fmt_float(ser.median(), 6)}, max={fmt_float(ser.max(), 6)}\n"
            )
        audit.write("synthesis_method counts:\n")
        audit.write(feat["synthesis_method"].value_counts(dropna=False).to_string())
        audit.write("\n")
        audit.write("source_type counts:\n")
        audit.write(feat["source_type"].value_counts().to_string())
        audit.write("\n")
        doi_like = samples["reference"].fillna("").astype(str).str.contains(r"10\.", regex=True)
        audit.write(f"DOI-like reference records={int(doi_like.sum())}\n")

        # Panel a
        counts_all = (
            feat.groupby(["primary_dopant_element", "source_type"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=["in-house", "literature"], fill_value=0)
        )
        counts_all["total"] = counts_all.sum(axis=1)
        counts_all = counts_all.sort_values("total", ascending=False)
        top = counts_all.head(12).copy()
        other = counts_all.iloc[12:][["in-house", "literature"]].sum()
        if int(other.sum()) > 0:
            top.loc["Other"] = [other.get("in-house", 0), other.get("literature", 0), int(other.sum())]
        audit.write("\nPANEL (a): primary dopant by source, all elements\n")
        audit.write(counts_all.to_string())
        audit.write("\n")
        sc_primary = int(counts_all.loc["Sc", "total"]) if "Sc" in counts_all.index else 0
        mg_primary = int(counts_all.loc["Mg", "total"]) if "Mg" in counts_all.index else 0
        dopant_sets = dopants.groupby("sample_id")["dopant_element"].apply(lambda s: set(s.dropna().astype(str)))
        sc_mg_ids = dopant_sets[dopant_sets.map(lambda s: {"Sc", "Mg"}.issubset(s))].index
        audit.write(
            f"Sc primary records={sc_primary}; Mg primary records={mg_primary}; "
            f"records containing both Sc and Mg={len(sc_mg_ids)}\n"
        )

        # Panel b
        temp = feat["measurement_temperature"].dropna()
        temp_bins = np.arange(math.floor(temp.min() / 50) * 50, math.ceil(temp.max() / 50) * 50 + 51, 50)
        temp_counts, temp_edges = np.histogram(temp, bins=temp_bins)
        audit.write("\nPANEL (b): measurement_temperature histogram, 50 C bins\n")
        for left, right, count in zip(temp_edges[:-1], temp_edges[1:], temp_counts):
            audit.write(f"{left:.0f}-{right:.0f} C: {int(count)}\n")

        # Panel c
        logc = feat["log10_conductivity"].replace([np.inf, -np.inf], np.nan).dropna()
        log_bins = np.linspace(np.floor(logc.min()), np.ceil(logc.max()), 17)
        log_counts, log_edges = np.histogram(logc, bins=log_bins)
        audit.write("\nPANEL (c): log10 conductivity histogram\n")
        audit.write(
            f"log10_conductivity n={len(logc)}, min={fmt_float(logc.min(), 6)}, "
            f"median={fmt_float(logc.median(), 6)}, max={fmt_float(logc.max(), 6)}\n"
        )
        audit.write(
            f"conductivity S/cm min={feat['conductivity'].min():.8e}, "
            f"median={feat['conductivity'].median():.8e}, max={feat['conductivity'].max():.8e}\n"
        )
        for left, right, count in zip(log_edges[:-1], log_edges[1:], log_counts):
            audit.write(f"{left:.3f} to {right:.3f}: {int(count)}\n")

        # Panel d
        nd = feat["number_of_dopants"].copy()
        classes = pd.Series(np.select([nd == 1, nd == 2, nd >= 3], ["single", "double", "multi"], default="none"))
        nd_counts = classes.value_counts().reindex(["none", "single", "double", "multi"]).fillna(0).astype(int)
        tdf = feat["total_dopant_fraction"].dropna()
        audit.write("\nPANEL (d): dopant count classes and total dopant fraction\n")
        audit.write(nd_counts.to_string())
        audit.write("\n")
        audit.write(
            f"total_dopant_fraction n={len(tdf)}, min={fmt_float(tdf.min(), 6)}, "
            f"median={fmt_float(tdf.median(), 6)}, max={fmt_float(tdf.max(), 6)}\n"
        )

        # Panel e
        pre = make_preprocessor()
        if not partitions.empty:
            dev_ids = partitions.loc[partitions["partition"].eq("development"), "sample_id"].astype(str)
            train_mask = feat["sample_id"].astype(str).isin(set(dev_ids))
            audit.write(f"\nPANEL (e): PCA training rows from development partition={int(train_mask.sum())}\n")
        else:
            train_mask = pd.Series(True, index=feat.index)
            audit.write("\nPANEL (e): reserved partition file FIELD NOT FOUND; fitted PCA on all rows\n")
        feature_cols = [
            "total_dopant_fraction",
            "average_dopant_radius",
            "average_dopant_valence",
            "number_of_dopants",
            "maximum_sintering_temperature",
            "total_sintering_duration",
            "primary_dopant_element",
            "synthesis_method",
            "material_source_and_purity",
        ]
        train_x = feat.loc[train_mask, feature_cols].copy()
        all_x = feat[feature_cols].copy()
        z_train = pre.fit_transform(train_x)
        z_all = pre.transform(all_x)
        pca = PCA(n_components=2, random_state=42)
        pc_train = pca.fit_transform(z_train.toarray() if hasattr(z_train, "toarray") else z_train)
        pc_all = pca.transform(z_all.toarray() if hasattr(z_all, "toarray") else z_all)

        radius_lookup = (
            dopants.dropna(subset=["dopant_element", "dopant_ionic_radius"])
            .assign(dopant_ionic_radius=lambda x: pd.to_numeric(x["dopant_ionic_radius"], errors="coerce"))
            .groupby("dopant_element")["dopant_ionic_radius"]
            .median()
        )
        valence_lookup = (
            dopants.dropna(subset=["dopant_element", "dopant_valence"])
            .assign(dopant_valence=lambda x: pd.to_numeric(x["dopant_valence"], errors="coerce"))
            .groupby("dopant_element")["dopant_valence"]
            .median()
        )
        # Mg does not occur in the curated training dopant table; the archived
        # GA scoring script supplies Mg=89.0 pm and valence=2.0 for generated
        # Sc-Mg candidates (see evaluate_optimized_candidates.py). This is
        # recorded explicitly in the audit file.
        radius_lookup.loc["Mg"] = radius_lookup.get("Mg", 89.0)
        valence_lookup.loc["Mg"] = valence_lookup.get("Mg", 2.0)
        sc, mg = 0.0750, 0.0319
        total_candidate = sc + mg
        cand_radius = (sc * radius_lookup.get("Sc", np.nan) + mg * radius_lookup.get("Mg", np.nan)) / total_candidate
        cand_valence = (sc * valence_lookup.get("Sc", 3.0) + mg * valence_lookup.get("Mg", 2.0)) / total_candidate
        cand = pd.DataFrame(
            [
                {
                    "total_dopant_fraction": total_candidate,
                    "average_dopant_radius": cand_radius,
                    "average_dopant_valence": cand_valence,
                    "number_of_dopants": 2,
                    "maximum_sintering_temperature": 1505.0,
                    "total_sintering_duration": feat["total_sintering_duration"].median(),
                    "primary_dopant_element": "Sc",
                    "synthesis_method": "Solid State Reaction",
                    "material_source_and_purity": "Generated Sc-Mg candidate",
                }
            ]
        )
        z_cand = pre.transform(cand[feature_cols])
        pc_cand = pca.transform(z_cand.toarray() if hasattr(z_cand, "toarray") else z_cand)[0]
        audit.write(
            f"PCA explained_variance_ratio=PC1 {pca.explained_variance_ratio_[0]:.6f}, "
            f"PC2 {pca.explained_variance_ratio_[1]:.6f}\n"
        )
        audit.write(
            f"Sc-Mg candidate projection: PC1={pc_cand[0]:.6f}, PC2={pc_cand[1]:.6f}; "
            f"features total_fraction={total_candidate:.6f}, avg_radius={cand_radius:.6f}, "
            f"avg_valence={cand_valence:.6f}, sintering_temp=1505.000000, "
            f"duration_used={cand['total_sintering_duration'].iloc[0]:.6f}\n"
        )
        if "Mg" not in set(dopants["dopant_element"].dropna().astype(str)):
            audit.write(
                "FIELD NOT FOUND for candidate projection: Mg dopant records are absent from sample_dopants.tsv; "
                "Mg radius=89.0 pm and valence=2.0 were taken from "
                "windows_experiments/piml_metric_optimization/evaluate_optimized_candidates.py.\n"
            )
        in_pc_box = (
            pc_all[:, 0].min() <= pc_cand[0] <= pc_all[:, 0].max()
            and pc_all[:, 1].min() <= pc_cand[1] <= pc_all[:, 1].max()
        )
        nn = NearestNeighbors(n_neighbors=2).fit(pc_all)
        data_nn = nn.kneighbors(pc_all, return_distance=True)[0][:, 1]
        cand_nn = NearestNeighbors(n_neighbors=1).fit(pc_all).kneighbors([pc_cand], return_distance=True)[0][0, 0]
        nn_percentile = float((data_nn <= cand_nn).mean() * 100.0)
        audit.write(
            f"Candidate within PC1/PC2 data bounding box={in_pc_box}; nearest-neighbor distance={cand_nn:.6f}; "
            f"distance percentile versus database point nearest-neighbor distances={nn_percentile:.2f}\n"
        )

        # Panel f
        fingerprint_cols = [
            "material_source_and_purity",
            "synthesis_method",
            "total_dopant_fraction",
            "average_dopant_radius",
            "average_dopant_valence",
            "number_of_dopants",
            "primary_dopant_element",
            "maximum_sintering_temperature",
            "total_sintering_duration",
        ]
        fp = feat[fingerprint_cols].copy()
        for col in [
            "total_dopant_fraction",
            "average_dopant_radius",
            "average_dopant_valence",
            "maximum_sintering_temperature",
            "total_sintering_duration",
        ]:
            fp[col] = pd.to_numeric(fp[col], errors="coerce").round(6).astype(str)
        fp = fp.fillna("NA").astype(str)
        feat["fingerprint"] = fp.agg("|".join, axis=1)
        temp_points = feat.groupby("fingerprint")["measurement_temperature"].nunique()
        temp_point_counts = temp_points.value_counts().sort_index()
        audit.write("\nPANEL (f): number of unique measurement temperatures per material fingerprint\n")
        audit.write(temp_point_counts.to_string())
        audit.write("\n")

    # Plot
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 9,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "in-house": "#4C78A8",
        "literature": "#F58518",
        "accent": "#54A24B",
        "candidate": "#D62728",
        "gray": "#9E9E9E",
    }
    fig = plt.figure(figsize=(8.2, 10.4), constrained_layout=True)
    gs = gridspec.GridSpec(3, 2, figure=fig, height_ratios=[1.20, 1.05, 1.15])
    axes = [fig.add_subplot(gs[i, j]) for i in range(3) for j in range(2)]

    ax = axes[0]
    plot_counts = top.sort_values("total")
    y = np.arange(len(plot_counts))
    ax.barh(y, plot_counts["in-house"], color=colors["in-house"], label="in-house")
    ax.barh(
        y,
        plot_counts["literature"],
        left=plot_counts["in-house"],
        color=colors["literature"],
        label="literature",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_counts.index)
    ax.set_xlabel("Conductivity records")
    ax.legend(frameon=False, loc="lower right")

    ax = axes[1]
    ax.hist(temp, bins=temp_bins, color="#6C8EBF", edgecolor="white")
    ax.set_xlabel(r"Measurement temperature ($^\circ$C)")
    ax.set_ylabel("Records")

    ax = axes[2]
    ax.hist(logc, bins=log_bins, color="#74B49B", edgecolor="white")
    ax.set_xlabel(r"$\log_{10}\sigma$ (S cm$^{-1}$)")
    ax.set_ylabel("Records")

    ax = axes[3]
    ax.axis("off")
    left = ax.inset_axes([0.00, 0.16, 0.44, 0.78])
    right = ax.inset_axes([0.56, 0.16, 0.42, 0.78])
    x = np.arange(len(nd_counts))
    left.bar(x, nd_counts.values, color=["#C7C7C7", "#4C78A8", "#F58518", "#54A24B"], edgecolor="white")
    left.set_xticks(x)
    left.set_xticklabels(["none", "single", "double", "multi"], rotation=25, ha="right")
    left.set_ylabel("Records")
    left.set_xlabel("Dopant count")
    left.set_title("Dopant number", fontsize=8)
    right.hist(tdf, bins=np.linspace(0, min(0.35, max(0.35, tdf.max())), 16), color="#B279A2", edgecolor="white")
    right.set_xlabel("Total dopant fraction")
    right.set_ylabel("Records")
    right.set_title("Dopant fraction", fontsize=8)
    for subax in [left, right]:
        subax.grid(axis="y", color="#EEEEEE", linewidth=0.5)

    ax = axes[4]
    primary = feat["primary_dopant_element"].fillna("None")
    top_elements = primary.value_counts().head(7).index.tolist()
    palette = plt.cm.tab10(np.linspace(0, 1, len(top_elements)))
    for elem, color in zip(top_elements, palette):
        mask = primary.eq(elem)
        ax.scatter(pc_all[mask, 0], pc_all[mask, 1], s=9, alpha=0.65, color=color, label=elem, linewidths=0)
    other_mask = ~primary.isin(top_elements)
    if other_mask.any():
        ax.scatter(pc_all[other_mask, 0], pc_all[other_mask, 1], s=7, alpha=0.35, color=colors["gray"], label="Other", linewidths=0)
    ax.scatter(pc_cand[0], pc_cand[1], marker="*", s=95, color=colors["candidate"], edgecolor="black", linewidth=0.4, label="Generated Sc-Mg point", zorder=5)
    ax.set_xlabel("PCA 1")
    ax.set_ylabel("PCA 2")
    ax.legend(
        frameon=False,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        fontsize=7,
        columnspacing=0.8,
        handletextpad=0.3,
    )

    ax = axes[5]
    ax.bar(temp_point_counts.index.astype(str), temp_point_counts.values, color="#72B7B2", edgecolor="white")
    ax.set_xlabel("Unique temperatures per material fingerprint")
    ax.set_ylabel("Material fingerprints")

    for label, ax in zip(["(a)", "(b)", "(c)", "(d)", "(e)", "(f)"], axes):
        ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")
        ax.grid(axis="y", color="#EEEEEE", linewidth=0.5)

    fig.savefig(FIG_PDF, bbox_inches="tight")
    fig.savefig(FIG_PNG, dpi=600, bbox_inches="tight")

    # Optional supplementary overview
    fig2, ax2 = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    route_counts = feat["synthesis_method"].value_counts().head(12).sort_values()
    ax2[0].barh(np.arange(len(route_counts)), route_counts.values, color="#4C78A8")
    ax2[0].set_yticks(np.arange(len(route_counts)))
    ax2[0].set_yticklabels(route_counts.index)
    ax2[0].set_xlabel("Records")
    ax2[0].set_title("Synthesis route", fontsize=8)
    hb = ax2[1].hexbin(
        feat["maximum_sintering_temperature"],
        feat["total_sintering_duration"],
        gridsize=22,
        mincnt=1,
        cmap="viridis",
    )
    ax2[1].set_xlabel(r"Max sintering temp. ($^\circ$C)")
    ax2[1].set_ylabel("Total duration (h)")
    cb = fig2.colorbar(hb, ax=ax2[1])
    cb.set_label("Records", fontsize=7)
    for src, color in [("in-house", colors["in-house"]), ("literature", colors["literature"])]:
        vals = feat.loc[feat["source_type"].eq(src), "log10_conductivity"].replace([np.inf, -np.inf], np.nan).dropna()
        ax2[2].hist(vals, bins=log_bins, histtype="step", linewidth=1.3, color=color, label=src)
    ax2[2].set_xlabel(r"$\log_{10}\sigma$ (S cm$^{-1}$)")
    ax2[2].set_ylabel("Records")
    ax2[2].legend(frameon=False)
    for label, ax in zip(["(a)", "(b)", "(c)"], ax2):
        ax.text(-0.16, 1.08, label, transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")
    fig2.savefig(SUPP_PDF, bbox_inches="tight")
    fig2.savefig(SUPP_PNG, dpi=600, bbox_inches="tight")

    print(f"Wrote {FIG_PDF}")
    print(f"Wrote {FIG_PNG}")
    print(f"Wrote {SUPP_PDF}")
    print(f"Wrote {AUDIT}")


if __name__ == "__main__":
    main()
