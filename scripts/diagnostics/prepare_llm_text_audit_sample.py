from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
DATA = ROOT / "windows_material_conductivity_training_handoff" / "training_project" / "material-conductivity-data-clean_reference" / "data"
OUT = PAPER / "results"


def source_type(ref):
    ref = "" if pd.isna(ref) else str(ref)
    return "literature" if ref.startswith("10.") or "/" in ref else "in_house_or_internal"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    raw = pd.read_csv(DATA / "raw_conductivity_samples.tsv", sep="\t")
    norm = pd.read_csv(DATA / "material_samples.tsv", sep="\t")
    dopants = pd.read_csv(DATA / "sample_dopants.tsv", sep="\t")

    dop_summary = (
        dopants.sort_values(["sample_id", "dopant_molar_fraction"], ascending=[True, False])
        .groupby("sample_id")
        .agg(
            primary_dopant=("dopant_element", "first"),
            dopant_system=("dopant_element", lambda s: "-".join(map(str, s))),
            total_dopant_fraction=("dopant_molar_fraction", "sum"),
        )
        .reset_index()
    )

    merged = raw.merge(
        norm[["sample_id", "material_source_and_purity", "synthesis_method", "processing_route"]],
        on="sample_id",
        suffixes=("_raw", "_normalized"),
        how="left",
    ).merge(dop_summary, on="sample_id", how="left")
    merged["source_type"] = merged["reference"].map(source_type)
    merged["stratum"] = (
        merged["source_type"].astype(str)
        + "|"
        + merged["primary_dopant"].fillna("unknown").astype(str)
        + "|"
        + merged["synthesis_method_normalized"].fillna("unknown").astype(str)
    )

    rng = np.random.default_rng(20260529)
    chosen = []
    for _, group in merged.groupby("stratum", sort=False):
        n_take = min(2, len(group))
        chosen.append(group.sample(n=n_take, random_state=int(rng.integers(0, 2**31 - 1))))
    sample = pd.concat(chosen).drop_duplicates("sample_id")
    if len(sample) < 150:
        remaining = merged[~merged["sample_id"].isin(sample["sample_id"])]
        extra = remaining.sample(n=min(150 - len(sample), len(remaining)), random_state=20260529)
        sample = pd.concat([sample, extra]).drop_duplicates("sample_id")
    sample = sample.head(150).copy()

    out = pd.DataFrame(
        {
            "sample_id": sample["sample_id"],
            "source_type": sample["source_type"],
            "reference": sample["reference"],
            "raw_material_source_and_purity": sample["material_source_and_purity_raw"],
            "normalized_material_source_and_purity": sample["material_source_and_purity_normalized"],
            "raw_synthesis_method": sample["synthesis_method_raw"],
            "normalized_synthesis_method": sample["synthesis_method_normalized"],
            "raw_processing_route": sample["processing_route_raw"],
            "normalized_processing_route": sample["processing_route_normalized"],
            "primary_dopant": sample["primary_dopant"],
            "dopant_system": sample["dopant_system"],
            "total_dopant_fraction": sample["total_dopant_fraction"],
            "raw_sintering_temperature": sample["sintering_temperature"],
            "raw_sintering_duration": sample["sintering_duration"],
            "operating_temperature": sample["operating_temperature"],
            "conductivity": sample["conductivity"],
            "audit_text_correct": "",
            "audit_synthesis_route_correct": "",
            "audit_source_purity_correct": "",
            "audit_dopant_parsing_correct": "",
            "audit_notes": "",
        }
    )
    out.to_csv(OUT / "llm_text_audit_sample.csv", index=False, encoding="utf-8-sig")
    print(f"Wrote {len(out)} rows to {OUT / 'llm_text_audit_sample.csv'}")


if __name__ == "__main__":
    main()
