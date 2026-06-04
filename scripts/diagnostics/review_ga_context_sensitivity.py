"""Scaffold for GA candidate context-sensitivity scoring.

This script requires the original ML environment with torch installed and the
trained PIML checkpoint/preprocessor available. It is intentionally not used to
generate manuscript numbers in the current environment.
"""

from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[3]
    report = Path(__file__).resolve().parents[1] / "missing_ga_scoring_assets.md"
    report.write_text(
        "\n".join(
            [
                "# Missing GA Context-Sensitivity Result",
                "",
                "The context-sensitivity diagnostic requires the original PyTorch scoring environment.",
                "The current local Python environments do not provide torch, so the trained PIML checkpoint cannot be loaded safely.",
                "",
                "Planned contexts for the retained Sc=7.50 mol%, Mg=3.19 mol% candidate:",
                "- original fixed generated-candidate text/source context",
                "- neutral empty or average text/source context",
                "- representative source/purity strings sampled from the training set",
                "- synthesis routes including Solid State Reaction, Hydrothermal and Sol-gel if present in training data",
                "",
                "No new GA context-sensitivity numbers were generated in this run.",
                f"Workspace root inspected: {root}",
            ]
        ),
        encoding="utf-8",
    )
    print(f"Wrote {report}")


if __name__ == "__main__":
    main()
