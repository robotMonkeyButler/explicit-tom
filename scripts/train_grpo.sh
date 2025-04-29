CUDA_VISIBLE_DEVICES=2 accelerate launch \
  --config_file /data/haofeiy2/explicit-tom/scripts/accelerate_config_grpo.yaml \
  --main_process_port 29511 \
  /data/haofeiy2/explicit-tom/scripts/train_grpo.py \
  --model_name /data/models/Qwen2.5-7B-Instruct \
  --policy_adapter_path /data/haofeiy2/sotopia-rl/sft_qwen25_7b_sft_round_1_bc_data_top_2/checkpoint-1500 \
  --reward_adapter_path /data/haofeiy2/explicit-tom/checkpoints/rm/rm_goal_w_relationship_social_rules_4_26/checkpoint-7200 \
  --learning_rate 5e-6 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --grpo_data_path /data/haofeiy2/explicit-tom/data/sotopia_pi_round1_qwen_sft_all_with_instruct_string.json \
  --template_path /data/haofeiy2/explicit-tom/evals/qwen2.5-7b.jinja \
  --num_grpo_epochs 2 \
  --num_generations 16 \
  --max_length 512 \
  --reward_funcs format tag \
  --use_lora_train_grpo \
  --output_dir /data/haofeiy2/explicit-tom/checkpoints/grpo/tom_test_0428

