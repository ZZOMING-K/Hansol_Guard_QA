python train.py \
--model_name_or_path "google/gemma-2-9b-it" \
--lora_r 16 \
--lora_alpha 32 \
--lora_dropout 0.05 \
--use_double_quant True \
--use_flash_attn False \
--use_4bit_quantization True \
--compute_dtype "bfloat16" \
--output_dir \
--per_device_train_batch_size 1 \
--per_device_eval_batch_size 1 \
--gradient_accumulation_steps 8 \
--learning_rate 2e-5 \
--max_steps 10 \
--num_train_epochs 1 \
--logging_steps 10 \
--logging_strategy "steps" \
--save_strategy "steps" \
--eval_strategy "steps" \
--save_total_limit 3 \
--eval_steps 100 \
--lr_scheduler_type "linear" \
--bf16 True \
--seed 42 \
--load_best_model_at_end True \
--metric_for_best_model "eval_loss" \
--push_to_hub True \
--report_to "wandb" \
--dataset_text_field "text" \
--optim "paged_adamw_32bit" \