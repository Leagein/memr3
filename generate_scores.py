import argparse
import json
import os

import pandas as pd


def _sanitize_tag(tag: str) -> str:
    return tag.replace("/", "-")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate summary scores from evaluation results.")
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to the evaluation metrics JSON file (overrides auto-generated filenames).",
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        default="results/",
        help="Folder containing evaluation metric files when inferring the path automatically.",
    )
    parser.add_argument(
        "--technique_type",
        choices=["memr3_zep", "memr3_rag"],
        default="memr3_zep",
        help="Technique type used to build the default metrics filename when --input_file is not provided.",
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=256,
        help="Chunk size used in memr3_rag filenames (only used when inferring the input file path).",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Top-k/limit used to build the default metrics filename when --input_file is not provided.",
    )
    parser.add_argument(
        "--model_tag",
        type=str,
        default="gpt4.1",
        help="Model tag used in default filenames when --input_file is not provided.",
    )
    parser.add_argument(
        "--max_iteration",
        type=int,
        default=None,
        help="Max iteration used in memr3_rag filenames (only used when inferring the input file path).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="Seed identifier used in default filenames when --input_file is not provided.",
    )
    return parser.parse_args()


def _resolve_input_path(args: argparse.Namespace) -> str:
    if args.input_file:
        return args.input_file

    model_tag = _sanitize_tag(args.model_tag)
    candidate_filenames = []
    if args.technique_type == "memr3_rag":
        if args.max_iteration is not None:
            candidate_filenames.append(
                f"evaluation_metrics_memr3_rag_{args.chunk_size}_k{args.top_k}_{model_tag}_{args.max_iteration}_seed{args.seed}.json"
            )
        candidate_filenames.append(
            f"evaluation_metrics_memr3_rag_{args.chunk_size}_k{args.top_k}_{model_tag}_seed{args.seed}.json"
        )
    else:
        if args.max_iteration is not None:
            candidate_filenames.append(
                f"evaluation_metrics_memr3_zep_limit{args.top_k}_{model_tag}_{args.max_iteration}_seed{args.seed}.json"
            )
        candidate_filenames.append(f"evaluation_metrics_memr3_zep_limit{args.top_k}_{model_tag}_seed{args.seed}.json")

    for filename in candidate_filenames:
        candidate_path = os.path.join(args.output_folder, filename)
        if os.path.exists(candidate_path):
            return candidate_path

    return os.path.join(args.output_folder, candidate_filenames[0])


def main() -> None:
    args = _parse_args()
    metrics_path = _resolve_input_path(args)

    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"Could not find metrics file at {metrics_path}")

    with open(metrics_path, "r") as f:
        data = json.load(f)

    all_items = []
    for key in data:
        all_items.extend(data[key])

    df = pd.DataFrame(all_items)

    # Convert category to numeric type
    df["category"] = pd.to_numeric(df["category"])

    # Calculate mean scores by category
    result = df.groupby("category").agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    # Add count of questions per category
    result["count"] = df.groupby("category").size()

    # Reorder categories to match expected display order
    # desired_order = [4, 1, 3, 2]
    desired_order = [1, 2, 3, 4]
    order = [cat for cat in desired_order if cat in result.index] + [cat for cat in result.index if cat not in desired_order]
    result = result.loc[order]

    print("Mean Scores Per Category:")
    print(result)

    overall_means = df.agg({"bleu_score": "mean", "f1_score": "mean", "llm_score": "mean"}).round(4)

    print("\nOverall Mean Scores:")
    print(overall_means)


if __name__ == "__main__":
    main()
