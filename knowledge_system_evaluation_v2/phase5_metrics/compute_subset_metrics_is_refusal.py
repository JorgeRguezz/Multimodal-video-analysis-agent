import argparse
import json
from pathlib import Path
from statistics import mean


DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent
    / "evaluated_datasets"
    / "community_qa_dataset_final_v3.json"
)

SYSTEMS = [
    "vanilla_base",
    "sota_base",
    "bm25",
    "vector_only",
    "asr_only",
    "vision_only",
    "graph_rag",
]


def get_metric_values(rows, system, metric):
    values = []
    for row in rows:
        value = row.get("ablations", {}).get(system, {}).get(metric)
        if value is not None:
            values.append(value)
    return values


def get_refusal_values(rows, system):
    values = []
    for row in rows:
        value = row.get("ablations", {}).get(system, {}).get("is_refusal")
        if value is not None:
            values.append(1.0 if value else 0.0)
    return values


def fmt(value):
    return "NA" if value is None else f"{value:.4f}"


def summarize(rows, systems):
    summary = []
    for system in systems:
        correctness = get_metric_values(rows, system, "correctness_score")
        faithfulness = get_metric_values(rows, system, "faithfulness")
        refusal = get_refusal_values(rows, system)

        summary.append(
            {
                "system": system,
                "n_correctness": len(correctness),
                "mean_correctness": mean(correctness) if correctness else None,
                "n_faithfulness": len(faithfulness),
                "mean_faithfulness": mean(faithfulness) if faithfulness else None,
                "n_refusal": len(refusal),
                "refusal_rate": mean(refusal) if refusal else None,
            }
        )
    return summary


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Compute metrics for all ablations on the subset where one selector "
            "system did not refuse."
        )
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selector-system", default="graph_rag", choices=SYSTEMS)
    parser.add_argument("--systems", nargs="+", default=SYSTEMS, choices=SYSTEMS)
    args = parser.parse_args()

    with args.input.open("r", encoding="utf-8") as f:
        records = json.load(f)

    subset = [
        row
        for row in records
        if row.get("ablations", {}).get(args.selector_system, {}).get("is_refusal")
        is False
    ]

    print(f"Input: {args.input}")
    print(f"Subset: {args.selector_system}.is_refusal == false")
    print(f"Subset size: {len(subset)} / {len(records)}")
    print()
    print(
        "system,n_correctness,mean_correctness,"
        "n_faithfulness,mean_faithfulness,n_refusal,refusal_rate"
    )
    for row in summarize(subset, args.systems):
        print(
            ",".join(
                [
                    row["system"],
                    str(row["n_correctness"]),
                    fmt(row["mean_correctness"]),
                    str(row["n_faithfulness"]),
                    fmt(row["mean_faithfulness"]),
                    str(row["n_refusal"]),
                    fmt(row["refusal_rate"]),
                ]
            )
        )


if __name__ == "__main__":
    main()
