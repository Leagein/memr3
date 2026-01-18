export OPENAI_API_KEY=
export ZEP_API_KEY=
export OPENAI_BASE_URL=https://api.openai.com/v1
export EMBEDDING_API_KEY=
export EMBEDDING_BASE_URL=https://api.openai.com/v1
export MODEL=gpt-4o-mini

TECHNIQUE_TYPE=${TECHNIQUE_TYPE:-memr3_zep}
TOP_K=${TOP_K:-15}
MODEL_TAG=${MODEL_TAG:-gpt4o}
MAX_ITERATION=${MAX_ITERATION:-5}
SEED=${SEED:-1} # not real seed
OUTPUT_FOLDER=${OUTPUT_FOLDER:-results/}
DATA_PATH=${DATA_PATH:-dataset/locomo10.json}
DATASET_TAG=${DATASET_TAG:-locomo10}

mkdir -p "${OUTPUT_FOLDER}"

python src/zep/zep_ingestion.py

python run_experiments.py \
  --top_k "${TOP_K}" \
  --model_tag "${MODEL_TAG}" \
  --seed "${SEED}" \
  --technique_type "${TECHNIQUE_TYPE}" \
  --max_iterations "${MAX_ITERATION}" \
  --output_folder "${OUTPUT_FOLDER}" \
  --data_path "${DATA_PATH}" \
  --dataset_tag "${DATASET_TAG}"

python evals.py \
  --input_file "${OUTPUT_FOLDER}/memr3_zep_results_limit${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json" \
  --output_file "${OUTPUT_FOLDER}/evaluation_metrics_memr3_zep_limit${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json"

python generate_scores.py \
  --input_file "${OUTPUT_FOLDER}/evaluation_metrics_memr3_zep_limit${TOP_K}_${MODEL_TAG}_${DATASET_TAG}_${MAX_ITERATION}_seed${SEED}.json"
