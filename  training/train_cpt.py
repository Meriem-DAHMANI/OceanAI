import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline, GenerationConfig
from peft import LoraConfig
from trl import SFTConfig, SFTTrainer
from random import randint


# Login into Hugging Face Hub
from huggingface_hub import login
login()

# Import the fine-tuning dataset
dataset = load_dataset("Meriem-DH/marine-dataset-cpt", split="train")
dataset = dataset.shuffle().select(range(419))

def format_cpt(sample):
    return {"text": sample["text"]}

dataset = dataset.map(format_cpt, remove_columns=dataset.column_names, batched=False)
dataset = dataset.train_test_split(test_size=0.2)
print(dataset["train"][0])

# Fine-tune Gemma using TRL and the SFTTrainer
# Hugging Face model id
model_id = "google/gemma-4-E2B-it"

# Check if GPU benefits from bfloat16
if torch.cuda.get_device_capability()[0] >= 8:
    torch_dtype = torch.bfloat16
else:
    torch_dtype = torch.float16

# Define model init arguments
model_kwargs = dict(
    dtype=torch_dtype,
    device_map="auto", 
)

# BitsAndBytesConfig: Enables 4-bit quantization to reduce model size/memory usage
model_kwargs["quantization_config"] = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=model_kwargs['dtype'],
    bnb_4bit_quant_storage=model_kwargs['dtype'],
)

# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-4-E2B-it")
peft_config = LoraConfig(
    lora_alpha=16,
    lora_dropout=0.05,
    r=8, 
    bias="none",
    target_modules="all-linear", 
    task_type="CAUSAL_LM",
    modules_to_save=["lm_head", "embed_tokens"], 
    ensure_weight_tying=True,
)

# define SFT configuration
args = SFTConfig(
    output_dir="gemma-4-ocean-cpt",
    max_length=265,
    num_train_epochs=1,
    per_device_train_batch_size=1,
    optim="adamw_torch_fused",
    logging_steps=10, 
    save_strategy="epoch",
    eval_strategy="epoch",
    learning_rate=5e-5,
    fp16=False,
    bf16=True,
    max_grad_norm=0.3,
    lr_scheduler_type="constant",
    push_to_hub=True,  
    hub_model_id="Meriem-DH/gemma-4-ocean-cpt",
    report_to="tensorboard",  
    dataset_text_field="text",
    packing=True,
    dataset_kwargs={
        "add_special_tokens": False,
        "append_concat_token": True, 
    }
)


# Create Trainer object
trainer = SFTTrainer(
    model=model,
    args=args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    processing_class=tokenizer,
)

# traning
trainer.train()

# save the final model again to the Hugging Face Hub
trainer.save_model()

# Test Model Inference
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch_dtype,
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_id)

config = GenerationConfig.from_pretrained(model_id)
config.max_new_tokens = 256

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

rand_idx = randint(0, len(dataset["test"]) - 1)
test_sample = dataset["test"][rand_idx]

prompt = test_sample["text"][:200]

outputs = pipe(prompt, generation_config=config)
print(f"Prompt:\n{prompt}")
print(f"Continuation:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

# save CPT adapter
trainer.save_model("gemma-4-ocean-cpt")
tokenizer.save_pretrained("gemma-4-ocean-cpt")

trainer.push_to_hub("Meriem-DH/gemma-4-ocean-cpt")
tokenizer.push_to_hub("Meriem-DH/gemma-4-ocean-cpt")

