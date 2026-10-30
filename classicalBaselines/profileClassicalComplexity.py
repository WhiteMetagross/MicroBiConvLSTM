"""Profile parameters, MACs, and efficiency of saved classical baselines.

The MAC convention matches the project's neural-model reports: weighted
multiply-accumulates are counted, while normalization, nonlinear functions,
comparisons, and additions are excluded. XGBoost therefore reports tree
comparisons instead of an artificial MAC count.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict, deque
from pathlib import Path
from statistics import mean, pstdev
from typing import Any, Dict, Iterable, List, Tuple

import joblib
import numpy as np


MODEL_ORDER = ("logistic_regression", "rbf_svm", "xgboost")
DATASET_ORDER = (
    "ucihar",
    "motionsense",
    "wisdm",
    "pamap2",
    "opportunity",
    "unimib",
    "skoda",
    "daphnet",
)


def numeric_size(value: Any) -> int:
    return int(np.asarray(value).size)


def profile_logistic(pipeline: Any) -> Dict[str, Any]:
    scaler = pipeline.named_steps["scaler"]
    classifier = pipeline.named_steps["classifier"]
    dimensions = int(classifier.coef_.shape[1])
    outputs = int(classifier.coef_.shape[0])
    predictive_parameters = numeric_size(classifier.coef_) + numeric_size(classifier.intercept_)
    auxiliary_state = numeric_size(scaler.mean_) + numeric_size(scaler.scale_)
    return {
        "predictiveParameters": predictive_parameters,
        "auxiliaryStateScalars": auxiliary_state,
        "totalStoredScalars": predictive_parameters + auxiliary_state,
        "macs": dimensions * outputs,
        "treeComparisons": None,
        "supportVectors": None,
        "trees": None,
        "treeNodes": None,
        "complexity": f"O(C*T*log(T) + {dimensions}*{outputs})",
        "complexityGeneral": "O(C*T*log(T) + D*K)",
    }


def profile_svm(pipeline: Any) -> Dict[str, Any]:
    scaler = pipeline.named_steps["scaler"]
    pca = pipeline.named_steps["pca"]
    classifier = pipeline.named_steps["classifier"]
    dimensions = int(pca.components_.shape[1])
    components = int(pca.components_.shape[0])
    support_vectors = int(classifier.support_vectors_.shape[0])
    classes = int(len(classifier.classes_))
    pairwise_outputs = classes * (classes - 1) // 2

    predictive_parameters = (
        numeric_size(pca.components_)
        + numeric_size(classifier.support_vectors_)
        + numeric_size(classifier.dual_coef_)
        + numeric_size(classifier.intercept_)
    )
    auxiliary_state = (
        numeric_size(scaler.mean_)
        + numeric_size(scaler.scale_)
        + numeric_size(pca.mean_)
    )
    # PCA projection, RBF support-vector dot products, and decision accumulation.
    macs = dimensions * components + support_vectors * components + support_vectors * (classes - 1)
    return {
        "predictiveParameters": predictive_parameters,
        "auxiliaryStateScalars": auxiliary_state,
        "totalStoredScalars": predictive_parameters + auxiliary_state,
        "macs": macs,
        "treeComparisons": None,
        "supportVectors": support_vectors,
        "trees": None,
        "treeNodes": None,
        "complexity": (
            f"O(C*T*log(T) + {dimensions}*{components} + "
            f"{support_vectors}*{components} + {support_vectors}*{classes - 1})"
        ),
        "complexityGeneral": "O(C*T*log(T) + D*P + S*P + S*(K-1))",
        "pairwiseClassifiers": pairwise_outputs,
    }


def tree_depths(frame: Any) -> Tuple[int, int]:
    """Return the number of trees and sum of maximum root-to-leaf depths."""
    depth_sum = 0
    tree_count = 0
    for _, tree in frame.groupby("Tree", sort=False):
        tree_count += 1
        children: Dict[str, List[str]] = defaultdict(list)
        for row in tree.itertuples(index=False):
            if row.Feature == "Leaf":
                continue
            children[str(row.ID)].extend([str(row.Yes), str(row.No)])
        root = str(tree.iloc[0]["ID"])
        queue = deque([(root, 0)])
        maximum = 0
        visited = set()
        while queue:
            node, depth = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            maximum = max(maximum, depth)
            for child in children.get(node, []):
                queue.append((child, depth + 1))
        depth_sum += maximum
    return tree_count, depth_sum


def profile_xgboost(classifier: Any) -> Dict[str, Any]:
    frame = classifier.get_booster().trees_to_dataframe()
    leaves = int((frame["Feature"] == "Leaf").sum())
    splits = int(len(frame) - leaves)
    trees, comparisons = tree_depths(frame)
    # A node is the conventional capacity unit for decision-tree ensembles.
    nodes = int(len(frame))
    return {
        "predictiveParameters": nodes,
        "auxiliaryStateScalars": 0,
        "totalStoredScalars": nodes,
        "macs": 0,
        "treeComparisons": comparisons,
        "supportVectors": None,
        "trees": trees,
        "treeNodes": nodes,
        "splitNodes": splits,
        "leafNodes": leaves,
        "complexity": f"O(C*T*log(T) + {trees}*depth), at most {comparisons} comparisons",
        "complexityGeneral": "O(C*T*log(T) + N_trees*depth)",
    }


def profile_model(model_name: str, estimator: Any) -> Dict[str, Any]:
    if model_name == "logistic_regression":
        return profile_logistic(estimator)
    if model_name == "rbf_svm":
        return profile_svm(estimator)
    if model_name == "xgboost":
        return profile_xgboost(estimator)
    raise ValueError(f"Unsupported model: {model_name}")


def aggregate(records: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["model"], record["dataset"])].append(record)

    rows = []
    fields = (
        "macroF1",
        "predictiveParameters",
        "auxiliaryStateScalars",
        "totalStoredScalars",
        "macs",
        "treeComparisons",
        "supportVectors",
        "trees",
        "treeNodes",
    )
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            members = groups.get((model, dataset), [])
            if not members:
                continue
            row: Dict[str, Any] = {
                "model": model,
                "dataset": dataset,
                "runs": len(members),
                "complexity": members[0]["complexity"],
                "complexityGeneral": members[0]["complexityGeneral"],
            }
            for field in fields:
                values = [member[field] for member in members if member.get(field) is not None]
                row[f"{field}Mean"] = mean(values) if values else None
                row[f"{field}Std"] = pstdev(values) if len(values) > 1 else (0.0 if values else None)
            f1_percent = 100.0 * row["macroF1Mean"]
            row["f1PerKParameters"] = f1_percent / (row["predictiveParametersMean"] / 1_000.0)
            row["f1PerMMacs"] = (
                f1_percent / (row["macsMean"] / 1_000_000.0)
                if row["macsMean"] and row["macsMean"] > 0
                else None
            )
            row["f1PerMComparisons"] = (
                f1_percent / (row["treeComparisonsMean"] / 1_000_000.0)
                if row["treeComparisonsMean"] and row["treeComparisonsMean"] > 0
                else None
            )
            rows.append(row)
    return rows


def format_number(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    return f"{value:,.{digits}f}"


def model_summaries(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    summaries = []
    for model in MODEL_ORDER:
        members = [row for row in rows if row["model"] == model]
        f1_percent = 100.0 * mean(row["macroF1Mean"] for row in members)
        parameters = mean(row["predictiveParametersMean"] for row in members)
        macs = mean(row["macsMean"] for row in members)
        comparisons = [row["treeComparisonsMean"] for row in members if row["treeComparisonsMean"]]
        comparison_mean = mean(comparisons) if comparisons else None
        summaries.append(
            {
                "model": model,
                "datasetCount": len(members),
                "macroF1Percent": f1_percent,
                "predictiveParametersMean": parameters,
                "macsMean": macs,
                "treeComparisonsMean": comparison_mean,
                "f1PerKParameters": f1_percent / (parameters / 1_000.0),
                "f1PerMMacs": f1_percent / (macs / 1_000_000.0) if macs else None,
                "f1PerMComparisons": (
                    f1_percent / (comparison_mean / 1_000_000.0)
                    if comparison_mean
                    else None
                ),
            }
        )
    return summaries


def write_markdown(
    path: Path,
    rows: List[Dict[str, Any]],
    summaries: List[Dict[str, Any]],
) -> None:
    lines = [
        "# Classical Baseline Complexity Results:",
        "",
        "Parameters are effective predictive values required by the fitted estimator. For logistic regression these are coefficients and intercepts. For PCA-RBF SVM they are PCA loadings, support vectors, dual coefficients, and intercepts. For XGBoost, decision-tree nodes are used as the conventional capacity unit. Scaler statistics and PCA centering vectors are reported separately as auxiliary state.",
        "",
        "MACs follow the project's pure weighted-MAC convention and exclude fixed feature extraction, normalization, nonlinear functions, and comparisons. XGBoost performs tree comparisons rather than dense MACs, so F1/M-MACs is not applicable and F1/M-comparisons is supplied instead.",
        "",
        "## Cross-Dataset Summary:",
        "",
        "The efficiency ratios below are calculated from the cross-dataset mean F1 and cross-dataset mean complexity, matching the aggregation used for the neural-model comparison.",
        "",
        "| Model | Mean Macro-F1 (%) | Mean Parameters | Mean MACs | F1/K-Params | F1/M-MACs | Mean Tree Comparisons | F1/M-Comparisons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summaries:
        lines.append(
            "| {model} | {f1} | {params} | {macs} | {f1kp} | {f1mm} | {comparisons} | {f1mc} |".format(
                model=row["model"],
                f1=format_number(row["macroF1Percent"]),
                params=format_number(row["predictiveParametersMean"], 0),
                macs=format_number(row["macsMean"], 0) if row["macsMean"] else "N/A",
                f1kp=format_number(row["f1PerKParameters"]),
                f1mm=format_number(row["f1PerMMacs"]),
                comparisons=format_number(row["treeComparisonsMean"], 0),
                f1mc=format_number(row["f1PerMComparisons"]),
            )
        )
    lines.extend(
        [
            "",
            "## Per-Dataset Results:",
            "",
        "| Model | Dataset | Macro-F1 (%) | Parameters | Aux. state | MACs | F1/K-Params | F1/M-MACs | Tree comparisons | F1/M-comparisons |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in rows:
        lines.append(
            "| {model} | {dataset} | {f1} | {params} | {aux} | {macs} | {f1kp} | {f1mm} | {comparisons} | {f1mc} |".format(
                model=row["model"],
                dataset=row["dataset"],
                f1=format_number(100.0 * row["macroF1Mean"]),
                params=format_number(row["predictiveParametersMean"], 0),
                aux=format_number(row["auxiliaryStateScalarsMean"], 0),
                macs=format_number(row["macsMean"], 0),
                f1kp=format_number(row["f1PerKParameters"]),
                f1mm=format_number(row["f1PerMMacs"]),
                comparisons=format_number(row["treeComparisonsMean"], 0),
                f1mc=format_number(row["f1PerMComparisons"]),
            )
        )
    lines.extend(
        [
            "",
            "## Complexity:",
            "",
            "Let `T` be window length, `C` the sensor channels, `D=21C` the engineered features, `K` the classes, `P` the retained PCA components, and `S` the support vectors.",
            "",
            "- Feature extraction shared by all models is `O(C*T*log(T))`, dominated by per-channel FFTs.",
            "- Logistic regression is `O(C*T*log(T) + D*K)` per window.",
            "- PCA-RBF SVM is `O(C*T*log(T) + D*P + S*P + S*(K-1))` per window.",
            "- XGBoost is `O(C*T*log(T) + N_trees*depth)` per window and is comparison-dominated.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        default="./results/classical_baselines/classical_master17_5seeds",
    )
    parser.add_argument("--output-stem", default="classical_complexity")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    f1_by_run = {}
    for group in summary["groups"]:
        model = group["model"]
        dataset = group["dataset"]
        for seed in group["seeds"]:
            result_path = run_dir / model / dataset / f"seed{seed}" / "results.json"
            result = json.loads(result_path.read_text(encoding="utf-8"))
            f1_by_run[(model, dataset, int(seed))] = float(result["macroF1"])

    records = []
    for model in MODEL_ORDER:
        for dataset in DATASET_ORDER:
            for model_path in sorted((run_dir / model / dataset).glob("seed*/*.joblib")):
                seed = int(model_path.parent.name.removeprefix("seed"))
                estimator = joblib.load(model_path)
                record = profile_model(model, estimator)
                record.update(
                    {
                        "model": model,
                        "dataset": dataset,
                        "seed": seed,
                        "macroF1": f1_by_run[(model, dataset, seed)],
                        "modelPath": str(model_path.resolve()),
                    }
                )
                records.append(record)

    rows = aggregate(records)
    summaries = model_summaries(rows)
    output_stem = run_dir / args.output_stem
    (output_stem.with_suffix(".json")).write_text(
        json.dumps(
            {"records": records, "aggregates": rows, "modelSummaries": summaries},
            indent=2,
        ),
        encoding="utf-8",
    )
    csv_fields = [key for key in rows[0].keys() if key not in {"complexity", "complexityGeneral"}]
    with output_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row.get(key) for key in csv_fields} for row in rows)
    write_markdown(output_stem.with_suffix(".md"), rows, summaries)
    print(f"Profiled {len(records)} estimators into {len(rows)} aggregate rows.")
    print(f"Results: {output_stem.with_suffix('.md').resolve()}")


if __name__ == "__main__":
    main()
