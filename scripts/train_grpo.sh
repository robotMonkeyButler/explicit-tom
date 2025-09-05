CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 accelerate launch \
  --config_file /data/disk0/explicit-tom/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /data/disk0/explicit-tom/scripts/train_grpo.py \
  --model_name /data/disk0/models/Qwen2.5-7B-Instruct \
  --reward_adapter_path /data/disk0/sotopia-rl/rm_goal_direct_0507/checkpoint-6800\
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --grpo_data_path /data/disk0/explicit-tom/data/sotopia_pi_tom_with_r1_prompt.json \
  --template_path /data/disk0/explicit-tom/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --num_generations 16 \
  --max_length 512 \
  --reward_funcs format tag \
  --use_lora_train_grpo \
  --output_dir /data/disk0/explicit-tom/checkpoints/grpo_tom_rm_goal_w_relationship_sft_r1