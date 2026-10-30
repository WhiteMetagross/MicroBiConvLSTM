"""Write machine-readable loss histories and publication-ready curve figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence


def saveLossCurveArtifacts(
    outputDir: str | Path,
    fileStem: str,
    trainLosses: Sequence[float],
    validationLosses: Sequence[float],
    *,
    modelName: str,
    datasetName: str,
    seed: int,
    bestEpoch: int,
    validationSource: str,
) -> dict[str, str]:
    """Save loss data as JSON/CSV and render a deterministic PNG curve."""
    if len(trainLosses) != len(validationLosses):
        raise ValueError("Training and validation loss histories must have equal lengths.")
    if not trainLosses:
        raise ValueError("At least one loss value is required to render a curve.")

    outputPath = Path(outputDir)
    outputPath.mkdir(parents=True, exist_ok=True)
    epochs = list(range(1, len(trainLosses) + 1))
    trainValues = [float(value) for value in trainLosses]
    validationValues = [float(value) for value in validationLosses]

    jsonPath = outputPath / f"{fileStem}LossHistory.json"
    csvPath = outputPath / f"{fileStem}LossHistory.csv"
    figurePath = outputPath / f"{fileStem}LossCurve.png"

    payload = {
        "schemaVersion": 1,
        "model": modelName,
        "dataset": datasetName,
        "seed": int(seed),
        "bestEpoch": int(bestEpoch),
        "epochsTrained": len(epochs),
        "validationSource": validationSource,
        "epochs": epochs,
        "trainLoss": trainValues,
        "validationLoss": validationValues,
    }
    jsonPath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    with csvPath.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["epoch", "trainLoss", "validationLoss"])
        writer.writerows(zip(epochs, trainValues, validationValues))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
    })
    figure, axis = plt.subplots(figsize=(7.2, 4.6), dpi=160)
    axis.plot(epochs, trainValues, color="#176B87", linewidth=2.0, label="Training loss")
    axis.plot(epochs, validationValues, color="#C84B31", linewidth=2.0, label="Validation loss")
    if 1 <= bestEpoch <= len(epochs):
        axis.axvline(
            bestEpoch,
            color="#4A4A4A",
            linestyle="--",
            linewidth=1.1,
            alpha=0.8,
            label=f"Best macro-F1 epoch ({bestEpoch})",
        )
    axis.set_title(f"{modelName} on {datasetName} (seed {seed})")
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Cross-entropy loss")
    axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    axis.legend(frameon=False)
    figure.tight_layout()
    figure.savefig(figurePath, bbox_inches="tight")
    plt.close(figure)

    return {
        "lossHistoryJson": str(jsonPath),
        "lossHistoryCsv": str(csvPath),
        "lossCurvePng": str(figurePath),
    }
