import os
import json
import pandas as pd
import torch


# Force DeepSpeed to use CUDA version 12.4 (as torch was compiled with 12.4)
os.environ["DS_CUDA_VERSION"] = "12.4"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from ab.nn.api import data  # Import the LEMUR data API

# ---------------------------------------------------------------------------
# Check for DeepSpeed installation
# ---------------------------------------------------------------------------
try:
    import deepspeed  # noqa: F401
except ImportError:
    print("DeepSpeed is not installed. Please install it via 'pip install deepspeed>=0.9.3' and re-run the script.")
    raise

# Clear CUDA cache to free memory
print("Clearing CUDA cache...")
torch.cuda.empty_cache()

# Set environment variable to help with memory fragmentation:
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# Enable TF32 for faster matrix multiplication on Ampere GPUs
print("Enabling TF32 for faster matrix multiplication...")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


# ---------------------------------------------------------------------------
# Step 1: Prepare Fine-Tuning Data Using the LEMUR API
# ---------------------------------------------------------------------------
print("Step 1: Preparing fine-tuning data...")
df = data(only_best_accuracy=False)

def create_example(row):
    prompt = (
        f"Task: {row.get('task', 'N/A')}\n"
        f"Dataset: {row.get('dataset', 'N/A')}\n"
        f"Metric: {row.get('metric', 'N/A')}\n"
        f"Epoch: {row.get('epoch', 'N/A')}\n"
        f"Hyperparameters: {row.get('prm', {})}\n"
        "Dataset Description: [Include Internet-sourced dataset details here]\n"
        "Provide the corresponding NN model code and predicted accuracy."
    )
    response = row.get('nn_code', 'No NN code available.')
    return {"prompt": prompt, "response": response}

examples = [create_example(row) for _, row in df.iterrows()]
split_idx = int(0.8 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

os.makedirs("data", exist_ok=True)
with open("data/lemur_train.json", "w", encoding="utf-8") as f:
    json.dump(train_examples, f, indent=2)
with open("data/lemur_val.json", "w", encoding="utf-8") as f:
    json.dump(val_examples, f, indent=2)

train_dataset = Dataset.from_dict({
    "prompt": [ex["prompt"] for ex in train_examples],
    "response": [ex["response"] for ex in train_examples]
})
val_dataset = Dataset.from_dict({
    "prompt": [ex["prompt"] for ex in val_examples],
    "response": [ex["response"] for ex in val_examples]
})

def preprocess_function(examples):
    combined = [f"{p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

# ---------------------------------------------------------------------------
# Step 2: Load the Distilled Model and Tokenizer
# ---------------------------------------------------------------------------
print("Step 2: Loading model and tokenizer...")
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto"
)
# Note: FlashAttention is not enabled here since the 'use_flash_attn' argument is unsupported.
# For FlashAttention integration, refer to your model’s documentation or apply necessary patches post-loading.

# ---------------------------------------------------------------------------
# Step 3: Set Up QLoRA Fine-Tuning Configuration (PEFT) and Enable Gradient Checkpointing
# ---------------------------------------------------------------------------
print("Step 3: Setting up QLoRA and enabling gradient checkpointing...")
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, lora_config)
model.gradient_checkpointing_enable()  # Enable gradient checkpointing
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---------------------------------------------------------------------------
# Step 4: Tokenize the Datasets
# ---------------------------------------------------------------------------
print("Step 4: Tokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------------------------------------------------------
# Step 5: Set Up DeepSpeed Configuration for Offloading and Memory Efficient Optimizer
# ---------------------------------------------------------------------------
print("Step 5: Creating DeepSpeed configuration...")
ds_config = {
    "zero_optimization": {
        "stage": 3,
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True
    },
    "bf16": {
        "enabled": True
    },
    "train_micro_batch_size_per_gpu": 1,
    "gradient_accumulation_steps": 2,
    "zero_allow_untested_optimizer": True
}

with open("ds_config.json", "w") as f:
    json.dump(ds_config, f, indent=2)

# ---------------------------------------------------------------------------
# Step 6: Set Up Training Arguments and Trainer with Memory-Saving Modifications
# ---------------------------------------------------------------------------
print("Step 6: Setting up training arguments and trainer...")
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    eval_strategy="epoch",                   # Use eval_strategy instead of evaluation_strategy
    learning_rate=2e-4,
    per_device_train_batch_size=1,           # Small batch size to reduce memory usage
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=2,           # Simulate a larger batch size via accumulation
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none",                        # Disable external logging (e.g., W&B)
    bf16=True,                               # Enable BF16 mixed precision training
    optim="adamw_bnb_8bit",                  # Use an 8-bit optimizer to reduce memory footprint
    deepspeed="./ds_config.json"             # Enable DeepSpeed for offloading and memory efficiency
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

# ---------------------------------------------------------------------------
# Start Fine-Tuning
# ---------------------------------------------------------------------------
print("Starting fine-tuning...")
trainer.train()

print("Saving model and tokenizer...")
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuning complete. Model saved to './fine_tuned_model'.")
