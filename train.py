from dataclasses import dataclass, field
from typing import Optional
import pandas as pd 
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments , BitsAndBytesConfig , HfArgumentParser, set_seed
from utils import create_dataset
from peft import LoraConfig

# Define and parse arguments.
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="google/gemma-2-9b-it")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.05)
    lora_target_modules: Optional[list[str]] = field(default_factory=lambda: ["q_proj", "o_proj", "k_proj", "v_proj", "gate_proj", "up_proj", "down_proj"])
    lora_task_type: Optional[str] = field(default="CAUSAL_LM")
    use_double_quant: Optional[bool] = field(default=True)
    bnb_4bit_compute_dtype: Optional[str] = field(default="bfloat16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    use_flash_attn: Optional[bool] = field(default=False)
    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=True)
    compute_dtype: Optional[str] = field(default="bfloat16")

@dataclass
class TrainingArguments:
    output_dir: str = field(default="./results/models")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(default=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    max_steps: int = field(default=-1)
    num_train_epochs: int = field(default=2.0)
    logging_steps: int = field(default=10)
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    eval_strategy : str = field(default="steps")
    save_total_limit: int = field(default=3)
    eval_steps: int = field(default=100)
    lr_scheduler_type: str = field(default="linear")
    fp16: bool = field(default=False)
    bf16: bool = field(default=True)
    seed: int = field(default=42)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: Optional[str] = field(default="eval_loss")
    push_to_hub: bool = field(default=True)
    report_to: Optional[str] = field(default="wandb")
    dataset_text_field: str = field(default = 'text')
    optim: str = field(default = 'paged_adamw_8bit')
    run_name : str = field(default = 'sft')


def main(model_args, training_args):
    
 # Set seed for reproducibility
    set_seed(training_args.seed)

    # quantization
    if model_args.use_4bit_quantization :
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=model_args.use_4bit_quantization,
            bnb_4bit_quant_type=model_args.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=model_args.bnb_4bit_compute_dtype,
            bnb_4bit_use_double_quant=model_args.use_double_quant,
        )
    elif model_args.use_8bit_quantization :
        bnb_config = BitsAndBytesConfig(
            load_in_8bit=model_args.use_8bit_quantization
        )
    else :
        bnb_config = None

    # model
    model =  AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path ,
        device_map = 'auto' ,
        quantization_config = bnb_config ,
        torch_dtype = model_args.compute_dtype , 
        attn_implementation='eager'
    )

    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    df = pd.read_csv('./data/prepro_train.csv')

    # datasets
    train_dataset, eval_dataset = create_dataset(
        df,
        tokenizer,
    )

    # lora
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        lora_dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        task_type=model_args.lora_task_type,
    )


 # trainingconfig
    trainingargs = SFTConfig(
        output_dir=training_args.output_dir,
        per_device_train_batch_size=training_args.per_device_train_batch_size,
        per_device_eval_batch_size=training_args.per_device_eval_batch_size,
        gradient_accumulation_steps=training_args.gradient_accumulation_steps,
        learning_rate=training_args.learning_rate,
        max_steps=training_args.max_steps,
        num_train_epochs=training_args.num_train_epochs,
        logging_steps=training_args.logging_steps,
        logging_strategy=training_args.logging_strategy,
        save_strategy=training_args.save_strategy,
        eval_strategy=training_args.eval_strategy,
        save_total_limit=training_args.save_total_limit,
        eval_steps=training_args.eval_steps,
        lr_scheduler_type=training_args.lr_scheduler_type,
        fp16=training_args.fp16,
        bf16=training_args.bf16,
        seed=training_args.seed,
        load_best_model_at_end=training_args.load_best_model_at_end,
        metric_for_best_model=training_args.metric_for_best_model,
        push_to_hub=training_args.push_to_hub,
        report_to=training_args.report_to,
        dataset_text_field=training_args.dataset_text_field,
        optim=training_args.optim,
        run_name=training_args.run_name
    )

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        args=trainingargs,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # train
    trainer.train()

    # model save
    trainer.save_model(training_args.output_dir)

    ADAPTER_MODEL = './results/lora_adapter'

    trainer.model.save_pretrained(ADAPTER_MODEL)
    tokenizer.save_pretrained(ADAPTER_MODEL)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, training_args)