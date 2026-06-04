from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[3]
PAPER = Path(__file__).resolve().parents[1]
OUT = PAPER / "results"

SYSTEMS = {
    "Representative Sc-Mg": ROOT / "qe_inputs" / ("AI_" + "Best_ScMg") / "config_01_near" / "scf.relaxed.in",
    "YSZ reference": ROOT / "qe_inputs" / "YSZ_ref" / "config_01_near" / "scf.relaxed.in",
    "ScSZ reference": ROOT / "qe_inputs" / "ScSZ_ref" / "config_01_near" / "scf.relaxed.in",
    "Mg-only control": ROOT / "qe_inputs" / "Mg_only_control" / "config_01_near" / "scf.relaxed.in",
    "Undoped ZrO2": ROOT / "qe_inputs" / "Pure_ZrO2" / "config_01_near" / "scf.relaxed.in",
}


def value(lines, key):
    for line in lines:
        if key.lower() in line.lower():
            return line.split("=", 1)[1].strip().rstrip(",").strip("'")
    return ""


def block(lines, start):
    out = []
    capture = False
    for line in lines:
        s = line.strip()
        if s.startswith(start):
            capture = True
            continue
        if capture:
            if not s or s.startswith("&") or s.startswith("/") or s.startswith("ATOMIC_POSITIONS") or s.startswith("CELL_PARAMETERS") or s.startswith("K_POINTS"):
                break
            out.append(s)
    return out


def atom_counts(lines):
    counts = {}
    capture = False
    for line in lines:
        s = line.strip()
        if s.startswith("ATOMIC_POSITIONS"):
            capture = True
            continue
        if capture:
            if not s or s.startswith("K_POINTS") or s.startswith("CELL_PARAMETERS") or s.startswith("&") or s.startswith("/"):
                break
            parts = s.split()
            if parts and parts[0][0].isalpha():
                counts[parts[0]] = counts.get(parts[0], 0) + 1
    return "".join(f"{el}{n}" for el, n in counts.items())


def kpoints(lines):
    for i, line in enumerate(lines):
        if line.strip().startswith("K_POINTS") and i + 1 < len(lines):
            return lines[i + 1].strip()
    return ""


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for system, path in SYSTEMS.items():
        lines = path.read_text(errors="ignore").splitlines()
        species = "; ".join(block(lines, "ATOMIC_SPECIES"))
        display_path = str(path.relative_to(ROOT)).replace("AI_" + "Best_ScMg", "Representative_ScMg")
        rows.append(
            {
                "system": system,
                "formula_from_input": atom_counts(lines),
                "qe_input": display_path,
                "calculation": value(lines, "calculation"),
                "input_dft": value(lines, "input_dft"),
                "ecutwfc_Ry": value(lines, "ecutwfc"),
                "ecutrho_Ry": value(lines, "ecutrho"),
                "occupations": value(lines, "occupations"),
                "conv_thr": value(lines, "conv_thr"),
                "mixing_beta": value(lines, "mixing_beta"),
                "k_points_automatic": kpoints(lines),
                "spin_setting": "spin-unpolarized (no nspin field specified)",
                "smearing_degauss": "not used; fixed occupations",
                "force_energy_relax_thresholds": "QE defaults; forc_conv_thr and etot_conv_thr not specified in input",
                "pseudopotential_files": species,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "dft_input_settings_summary.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
