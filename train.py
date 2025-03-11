from dataclasses import dataclass, field
from typing import Optional
import pandas as pd 
from transformers import HfArgumentParser, set_seed
from trl import SFTConfig, SFTTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments , BitsAndBytesConfig 
from utils import create_datasets 
from peft import LoraConfig

# Define and parse arguments.
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default="rtzr/ko-gemma-2-9b-it")
    lora_alpha: Optional[int] = field(default=16)
    lora_dropout: Optional[float] = field(default=0.1)
    lora_r: Optional[int] = field(default=8)
    lora_target_modules: Optional[str] = field(default="q_proj,k_proj,v_proj,o_proj,down_proj,up_proj,gate_proj")
    lora_task_type: Optional[str] = field(default="CAUSAL_LM")
    use_double_quant: Optional[bool] = field(default=False)
    bnb_4bit_compute_dtype: Optional[str] = field(default="float16")
    bnb_4bit_quant_type: Optional[str] = field(default="nf4")
    use_flash_attn: Optional[bool] = field(default=False)
    use_peft_lora: Optional[bool] = field(default=False)
    use_8bit_quantization: Optional[bool] = field(default=False)
    use_4bit_quantization: Optional[bool] = field(default=False) 
    compute_dtype: Optional[str] = field(default="float16")

@dataclass
class TrainingArguments:
    output_dir: str = field(default="./results/models")
    per_device_train_batch_size: int = field(default=1)
    per_device_eval_batch_size: int = field(defaust=1)
    gradient_accumulation_steps: int = field(default=8)
    learning_rate: float = field(default=2e-5)
    max_steps: int = field(default=-1)
    num_train_epochs: int = field(default=2.0)
    logging_steps: int = field(default=10)
    logging_strategy: str = field(default="steps")
    save_strategy: str = field(default="steps")
    save_total_limit: int = field(default=2)
    eval_steps: int = field(default=100)
    lr_scheduler_type: str = field(default="linear")
    fp16: bool = field(default=True)  
    bf16: bool = field(default=False)
    seed: int = field(default=42)
    load_best_model_at_end: bool = field(default=True)
    metric_for_best_model: Optional[str] = field(default="eval_loss")
    push_to_hub: bool = field(default=True)
    report_to: Optional[str] = field(default="wandb") 
    max_seq_length: int = field(default=512)
    dataset_text_file: str = field(default = 'text')
    optim: str = field(default = 'adamw')


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
    model = AutoModelForCausalLM( 
        model_args.model_name_or_path , 
        device_map = 'auto' , 
        quantization_config = bnb_config , 
        torch_type = model_args.compute_dtype
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model)
    
    df = pd.read_csv('./data/train.csv')
    
    # datasets
    train_dataset, eval_dataset = create_datasets(
        df,
        tokenizer,
    )
    
    peft_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=model_args.lora_alpha,
        dropout=model_args.lora_dropout,
        target_modules=model_args.lora_target_modules,
        task_type = model_args.lora_task_type,
    )

    # trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        model_args=SFTConfig(training_args),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )
    
    trainer.train()
    trainer.save_model(training_args.output_dir) #모델 저장 
    
    ADAPTER_MODEL = './results/lora_adapter'
    
    trainer.model.save_pretrained(ADAPTER_MODEL)
    tokenizer.save_pretrained(ADAPTER_MODEL)

if __name__ == "__main__":
    parser = HfArgumentParser((ModelArguments, TrainingArguments))
    model_args, training_args = parser.parse_args_into_dataclasses()
    main(model_args, training_args)