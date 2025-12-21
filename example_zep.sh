export OPENAI_API_KEY=
export ZEP_API_KEY=
export OPENAI_BASE_URL=https://api.openai.com/v1
export MODEL=gpt-4o-mini
export MEMR3_RAG_CACHE_DIR=rag_cache-4o

TECHNIQUE_TYPE=${TECHNIQUE_TYPE:-memr3_rag}
CHUNK_SIZE=${CHUNK_SIZE:-256}
TOP_K=${TOP_K:-15}
MODEL_TAG=${MODEL_TAG:-gpt4o}
MAX_ITERATION=${MAX_ITERATION:-5}
SEED=${SEED:-1} # not real seed
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/}

mkdir -p results/

python src/zep/zep_ingestion.py

python run_experiments.py \
  --top_k "${TOP_K}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --technique_type memr3_zep \
  --max_iteration "${MAX_ITERATION}" \
  --output_folder "${OUTPUT_FOLDER}"

python evals.py \
  --input_file "results/memr3_zep_results_limit${TOP_K}_${MODEL_TAG}_${MAX_ITERATION}_seed${SEED}.json" \
  --output_file "results/evaluation_metrics_memr3_zep_limit${TOP_K}_${MODEL_TAG}_${MAX_ITERATION}_seed${SEED}.json"

python generate_scores.py \
  --technique_type "memr3_zep" \
  --top_k "${TOP_K}" \
  --max_iteration "${MAX_ITERATION}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}"