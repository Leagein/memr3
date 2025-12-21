export OPENAI_API_KEY=
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL=gpt-4o-mini
export MEMR3_RAG_CACHE_DIR=rag_cache-4o
export EMBEDDING_MODEL=text-embedding-3-large
export MEMR3_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2

TECHNIQUE_TYPE=${TECHNIQUE_TYPE:-memr3_rag}
CHUNK_SIZE=${CHUNK_SIZE:-256}
TOP_K=${TOP_K:-5}
MODEL_TAG=${MODEL_TAG:-gpt4o}
MAX_ITERATION=${MAX_ITERATION:-5}
SEED=${SEED:-1} # not real seed
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/}

mkdir -p results/

python run_experiments.py \
  --chunk_size "${CHUNK_SIZE}" \
  --top_k "${TOP_K}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --technique_type memr3_rag \
  --max_iteration "${MAX_ITERATION}" \
  --output_folder "${OUTPUT_FOLDER}"

python evals.py \
  --input_file "results/memr3_results_${CHUNK_SIZE}_k${TOP_K}_${MODEL_TAG}_${MAX_ITERATION}_seed${SEED}.json" \
  --output_file "results/evaluation_metrics_${TECHNIQUE_TYPE}_${CHUNK_SIZE}_k${TOP_K}_${MODEL_TAG}_${MAX_ITERATION}_seed${SEED}.json"

python generate_scores.py \
  --technique_type "${TECHNIQUE_TYPE}" \
  --chunk_size "${CHUNK_SIZE}" \
  --top_k "${TOP_K}" \
  --max_iteration "${MAX_ITERATION}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}"
