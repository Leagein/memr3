export OPENAI_API_KEY=
export OPENAI_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export MODEL=gpt-4o-mini
export MEMR3_RAG_CACHE_DIR=rag_longmemeval_cache
export EMBEDDING_MODEL=text-embedding-3-large
export MEMR3_RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-12-v2

TECHNIQUE_TYPE=${TECHNIQUE_TYPE:-memr3_rag}
CHUNK_SIZE=${CHUNK_SIZE:-256}
TOP_K=${TOP_K:-5}
MODEL_TAG=${MODEL_TAG:-gpt4o}
MAX_ITERATION=${MAX_ITERATION:-5}
SEED=${SEED:-1} # not real seed
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/}
DATA_PATH=${DATA_PATH:-dataset/longmemeval_s_cleaned.json}
DATASET_TAG=${DATASET_TAG:-longmemeval_s_cleaned}

mkdir -p "${OUTPUT_FOLDER}"

python run_experiments.py \
  --chunk_size "${CHUNK_SIZE}" \
  --top_k "${TOP_K}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --technique_type "${TECHNIQUE_TYPE}" \
  --max_iterations "${MAX_ITERATION}" \
  --output_folder "${OUTPUT_FOLDER}" \
  --data_path "${DATA_PATH}" \
  --dataset_tag "${DATASET_TAG}"

python evals.py \
  --input_file "${OUTPUT_FOLDER}/memr3_results_${CHUNK_SIZE}_k${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json" \
  --output_file "${OUTPUT_FOLDER}/evaluation_metrics_${TECHNIQUE_TYPE}_${CHUNK_SIZE}_k${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json"

python generate_scores.py \
  --input_file "${OUTPUT_FOLDER}/evaluation_metrics_${TECHNIQUE_TYPE}_${CHUNK_SIZE}_k${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json"
