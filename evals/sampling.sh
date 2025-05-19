export VLLM_GPU=0
export DJANGO_GPU=2
export VLLM_PORT=6010
export DJANGO_PORT=6020
export SFT_MODEL_FOLDER_NAME="sft_qwen25_7b_sft_round_1_bc_data_top_2"
export SFT_MODEL_CKPT_STEP=1500
export GRPO_MODEL_FOLDER_NAME="grpo_tom_rm_goal_w_relationship_sft_r1"
export GRPO_MODEL_CKPT_STEP=2250
export REPO_FOLDER_NAME="/data/haofeiy2/explicit-tom"
export SFT_MODEL_PATH="${REPO_FOLDER_NAME}/checkpoints/${SFT_MODEL_FOLDER_NAME}/checkpoint-${SFT_MODEL_CKPT_STEP}"
export GRPO_MODEL_PATH="${REPO_FOLDER_NAME}/checkpoints/${GRPO_MODEL_FOLDER_NAME}/checkpoint-${GRPO_MODEL_CKPT_STEP}"
export ENV_MODEL="gpt-4o"

export TAG="${GRPO_MODEL_FOLDER_NAME}_steps_${SFT_MODEL_CKPT_STEP}_vs_${SFT_MODEL_FOLDER_NAME}_steps_${GRPO_MODEL_CKPT_STEP}"
export SFT_MODEL_NAME="sft_qwen25_7b_sft_round_1_bc_data_top_2-gpu${VLLM_GPU}"
export MODEL_A=custom/${GRPO_MODEL_FOLDER_NAME}@http://localhost:${DJANGO_PORT}/sotopia
export MODEL_B=custom/${SFT_MODEL_NAME}@http://localhost:${VLLM_PORT}/v1
export REDIS_OM_URL="redis://:QzmCUD3C3RdsR@35.232.108.130:6379"
export SFT_MODEL_VLLM_API_URL="http://localhost:${VLLM_PORT}/v1/completions"


# Command 1: Launch the VLLM API server with LoRA enabled.
CUDA_VISIBLE_DEVICES=$VLLM_GPU python -m vllm.entrypoints.openai.api_server \
    --model /mnt/data_from_server1/models/Qwen2.5-7B-Instruct \
    --port "$VLLM_PORT" \
    --chat-template /data/haofeiy2/explicit-tom/evals/qwen2.5-7b.jinja \
    --served-model-name qwen25-7b-instruct \
    --enable-lora \
    --lora-modules "$SFT_MODEL_NAME=$SFT_MODEL_PATH"

# Command 2: Start the Django server with the specified configuration.
CUDA_VISIBLE_DEVICES=$DJANGO_GPU python /data/haofeiy2/explicit-tom/serves/manage.py start_with_config \
    --grpo_model_path "$GRPO_MODEL_PATH" \
    --base_model_path "/mnt/data_from_server1/models/Qwen2.5-7B-Instruct" \
    --template_path "/data/haofeiy2/explicit-tom/evals/qwen2.5-7b.jinja" \
    --port "$DJANGO_PORT" \
    --log_path "/data/haofeiy2/explicit-tom/evals/logs/$GRPO_MODEL_FOLDER_NAME"

# Command 3: Run experiment evaluations.
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=1 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT1_MODEL='${MODEL_A}'" \
  "--gin.AGENT2_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'" \
&& \
python examples/experiment_eval.py \
  --gin_file sotopia_conf/generation_utils_conf/generate.gin \
  --gin_file sotopia_conf/server_conf/server.gin \
  --gin_file sotopia_conf/run_async_server_in_batch.gin \
  --gin.BATCH_SIZE=1 \
  --gin.PUSH_TO_DB=True \
  '--gin.ENV_IDS=["01H7VFHNV13MHN97GAH73E3KM8", "01H7VFHN5WVC5HKKVBHZBA553R", "01H7VFHN9W0WAFZCBT09PKJJNK", "01H7VFHPDZVVCDZR3AARA547CY", "01H7VFHPQQQY6H4DNC6NBQ8XTG", "01H7VFHN7WJK7VWVRZZTQ6DX9T", "01H7VFHPS5WJW2694R1MNC8JFY", "01H7VFHNN7XTR99319DS8KZCQM", "01H7VFHQ11NAMZS4A2RDGDB01V", "01H7VFHPSWGDGEYRP63H2DJKV0", "01H7VFHNF4G18PC9JHGRC8A1R6", "01H7VFHNNYH3W0VRWVY178K2TK", "01H7VFHP8AN5643B0NR0NP00VE", "01H7VFHN7A1ZX5KSMT2YN9RXC4"]' \
  "--gin.ENV_MODEL='${ENV_MODEL}'" \
  "--gin.AGENT2_MODEL='${MODEL_A}'" \
  "--gin.AGENT1_MODEL='${MODEL_B}'" \
  "--gin.TAG='${TAG}'"