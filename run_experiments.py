import argparse
import os

from src.utils import METHODS, TECHNIQUES
from src.memr3_rag import MemR3RAGManager
from src.memr3_zep import MemR3ZepManager

def _sanitize_tag(tag: str) -> str:
    return tag.replace("/", "-")


class Experiment:
    def __init__(self, technique_type, chunk_size):
        self.technique_type = technique_type
        self.chunk_size = chunk_size

    def run(self):
        print(f"Running experiment with technique: {self.technique_type}, chunk size: {self.chunk_size}")


def main():
    parser = argparse.ArgumentParser(description="Run memory experiments")
    parser.add_argument("--technique_type", choices=TECHNIQUES, default="mem0", help="Memory technique to use")
    parser.add_argument("--method", choices=METHODS, default="add", help="Method to use")
    parser.add_argument("--chunk_size", type=int, default=1000, help="Chunk size for processing")
    parser.add_argument("--output_folder", type=str, default="results/", help="Output path for results")
    parser.add_argument("--top_k", type=int, default=30, help="Number of top memories to retrieve")
    parser.add_argument("--filter_memories", action="store_true", default=False, help="Whether to filter memories")
    parser.add_argument("--is_graph", action="store_true", default=False, help="Whether to use graph-based search")
    parser.add_argument("--model_tag", type=str, default="gpt4.1", help="Model tag used in output filenames")
    parser.add_argument("--seed", type=int, default=1, help="Seed identifier used in output filenames")
    parser.add_argument("--max_iterations", "--max_iteration", dest="max_iterations", type=int, default=5, help="Maximum reasoning iterations for the memr3 workflow")
    parser.add_argument("--data_path", type=str, default=None, help="Path to the dataset file (overrides per-technique default).")
    parser.add_argument("--dataset_tag", type=str, default=None, help="Optional dataset tag included in output filenames.")
    parser.add_argument("--run_id", type=str, default=os.getenv("ZEP_RUN_ID", "1"), help="Run identifier to align Zep add/search/memr3 datasets")

    args = parser.parse_args()

    # Add your experiment logic here
    print(f"Running experiments with technique: {args.technique_type}, chunk size: {args.chunk_size}")
    model_tag = _sanitize_tag(args.model_tag)
    dataset_tag = _sanitize_tag(args.dataset_tag) if args.dataset_tag else None
    tag_suffix = f"_{dataset_tag}" if dataset_tag else ""

    if args.technique_type == "memr3_rag":
        output_file_path = os.path.join(
            args.output_folder,
            f"memr3_results_{args.chunk_size}_k{args.top_k}_{model_tag}{tag_suffix}_{args.max_iterations}_seed{args.seed}.json",
        )
        data_path = args.data_path or "dataset/locomo10_rag.json"
        memr3_manager = MemR3RAGManager(
            data_path=data_path,
            chunk_size=args.chunk_size,
            top_k=args.top_k,
            max_iterations=args.max_iterations,
        )
        memr3_manager.process_all_conversations(output_file_path)
    elif args.technique_type == "memr3_zep":
        output_file_path = os.path.join(
            args.output_folder,
            f"memr3_zep_results_limit{args.top_k}_{model_tag}{tag_suffix}_{args.max_iterations}_seed{args.seed}.json",
        )
        data_path = args.data_path or "dataset/locomo10.json"
        memr3_manager = MemR3ZepManager(
            data_path=data_path,
            search_limit=args.top_k,
            max_iterations=args.max_iterations,
            run_id=args.run_id,
        )
        memr3_manager.process_all_conversations(output_file_path)


if __name__ == "__main__":
    main()
