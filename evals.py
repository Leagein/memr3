import argparse
import concurrent.futures
import json
import os
import threading
from collections import defaultdict

from metrics.llm_judge import evaluate_llm_judge
from metrics.utils import calculate_bleu_scores, calculate_metrics
from tqdm import tqdm


def build_item_identifier(key, item):
    return (str(key), str(item.get("question")), str(item.get("answer")))


def load_existing_results(path):
    results = defaultdict(list)
    processed_ids = set()

    if not os.path.exists(path):
        return results, processed_ids

    with open(path, "r") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            # Treat a corrupted or empty file as an empty result set.
            return results, processed_ids

    for k, items in data.items():
        for item in items:
            identifier = build_item_identifier(k, item)
            processed_ids.add(identifier)
            results[k].append(item)

    return results, processed_ids


def save_results(results, path):
    with open(path, "w") as f:
        json.dump(dict(results), f, indent=4)


def process_item(item_data):
    key, item = item_data

    gt_answer = str(item["answer"])
    pred_answer = str(item["response"])
    category = str(item["category"])
    question = str(item["question"])

    # Skip category 5
    if category == "5":
        return key, None

    metrics = calculate_metrics(pred_answer, gt_answer)
    bleu_scores = calculate_bleu_scores(pred_answer, gt_answer)
    llm_score = evaluate_llm_judge(question, gt_answer, pred_answer)

    return (
        key,
        {
            "question": question,
            "answer": gt_answer,
            "response": pred_answer,
            "category": category,
            "bleu_score": bleu_scores["bleu1"],
            "f1_score": metrics["f1"],
            "llm_score": llm_score,
        },
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate RAG results")
    parser.add_argument(
        "--input_file", type=str, default="results/rag_results_500_k1.json", help="Path to the input dataset file"
    )
    parser.add_argument(
        "--output_file", type=str, default="evaluation_metrics.json", help="Path to save the evaluation results"
    )
    parser.add_argument("--max_workers", type=int, default=10, help="Maximum number of worker threads")

    args = parser.parse_args()

    with open(args.input_file, "r") as f:
        data = json.load(f)

    results, processed_ids = load_existing_results(args.output_file)

    items_to_process = []
    for key, entries in data.items():
        for item in entries:
            identifier = build_item_identifier(key, item)
            if identifier in processed_ids:
                continue
            items_to_process.append((key, item))

    if not items_to_process:
        print(f"All items from {args.input_file} already exist in {args.output_file}. Nothing to do.")
        return
    results_lock = threading.Lock()

    # Use ThreadPoolExecutor with specified workers
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(process_item, item_data) for item_data in items_to_process]

        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            key, processed_item = future.result()

            if processed_item is None:
                continue

            identifier = build_item_identifier(key, processed_item)

            with results_lock:
                if identifier in processed_ids:
                    continue

                results[key].append(processed_item)
                processed_ids.add(identifier)
                save_results(results, args.output_file)

    print(f"Results saved to {args.output_file}")


if __name__ == "__main__":
    main()
