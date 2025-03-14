#!/usr/bin/env python
"""
Fine-Tuning Pipeline for DeepSeek-R1-Distill-Qwen-1.5B

This script fine-tunes a distilled DeepSeek model using examples derived from the LEMUR dataset.
Each training example is structured as a prompt that includes task, dataset, metric, hyperparameters,
and a placeholder for an external dataset description. The response combines the NN model code with 
performance metrics (accuracy and epoch).

DeepSpeed is used for offloading to CPU and memory efficiency, and QLoRA via PEFT is used to fine-tune
only a small subset of model parameters.

If a fine-tuned model already exists in the "./fine_tuned_model" directory, the script loads that model
to avoid re-training.
"""

# ========================== All Imports ==========================
import os
import json
import pandas as pd
import torch

# Set environment variables for DeepSpeed and CUDA memory management
os.environ["DS_CUDA_VERSION"] = "12.4"
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model
from datasets import Dataset
from ab.nn.api import data  # LEMUR data API
from setup_data import run_setup
from config.config import FINE_TUNED_MODEL_DIR, HF_TOKEN  # HF_TOKEN from your environment or config

# ========================== Constants ==========================
MAX_LENGTH = 2048  # Maximum tokens for training examples

# ========================== Helper Functions ==========================
def create_example(row):
    """
    Converts a row from the LEMUR dataset into a training example.
    The prompt contains task, dataset, metric, hyperparameters, and a placeholder for an external dataset description.
    The response combines the NN model code with performance metrics (predicted accuracy and epoch).
    """
    prompt = (
        f"Task: {row.get('task', 'N/A')}\n"
        f"Dataset: {row.get('dataset', 'N/A')}\n"
        f"Metric: {row.get('metric', 'N/A')}\n"
        f"Hyperparameters: {row.get('prm', {})}\n"
        "Dataset Description: [Insert external dataset description here]\n"
        "Based on the above, provide the corresponding NN model code along with predicted accuracy and epoch."
    )
    nn_code = row.get('nn_code', 'No NN code available.')
    accuracy = row.get('accuracy', 'N/A')
    epoch = row.get('epoch', 'N/A')
    response = f"NN Code:\n{nn_code}\nPredicted Accuracy: {accuracy}\nEpoch: {epoch}"
    return {"prompt": prompt, "response": response}

def preprocess_function(examples):
    """
    Combines prompt and response into a single text and tokenizes it.
    The tokenized output is truncated/padded to MAX_LENGTH.
    """
    combined = [f"{p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(combined, truncation=True, padding="max_length", max_length=MAX_LENGTH)
    tokenized["labels"] = tokenized["input_ids"].copy()  # For causal language modeling
    return tokenized

# ========================== Main Fine-Tuning Function ==========================
def main():
    # Check if a fine-tuned model already exists; if so, skip training to save resources.
    if os.path.isdir(FINE_TUNED_MODEL_DIR):
        print(f"Fine-tuned model already exists at {FINE_TUNED_MODEL_DIR}. Skipping training.")
        return

    # Step 0: Run setup to ensure external data sources are available.
    # The purpose of run_setup() is to clone required GitHub repositories and create dataset description files.
    # This external data is critical for our subsequent retrieval-augmented generation pipeline.
    print("Running setup to ensure external data sources are available...")
    run_setup()

    # Step 1: Prepare training data using the LEMUR API.
    print("Preparing fine-tuning data from LEMUR API...")
    df = data(only_best_accuracy=False)
    
    examples = [create_example(row) for _, row in df.iterrows()]
    split_idx = int(0.8 * len(examples))
    train_examples = examples[:split_idx]
    val_examples = examples[split_idx:]

    # Save examples for inspection.
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


    # Step 2: Load the base model and tokenizer (DeepSeek-R1-Distill-Qwen-1.5B)
    print("Loading model and tokenizer...")
    MODEL_ID_local = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    global tokenizer  # so that preprocess_function can access it
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_local, trust_remote_code=True, use_auth_token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_local,
        trust_remote_code=True,
        torch_dtype="auto",
        use_auth_token=HF_TOKEN
    )

    # Step 3: Set up QLoRA configuration (PEFT) and enable gradient checkpointing.
    print("Setting up QLoRA and enabling gradient checkpointing...")
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()  # Saves memory during backpropagation.
    model.enable_input_require_grads()       # Ensure inputs require gradients.
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # Step 4: Tokenize the datasets.
    print("Tokenizing datasets...")
    tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
    tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Step 5: Create DeepSpeed configuration for offloading and memory efficiency.
    print("Creating DeepSpeed configuration...")
    ds_config = {
        "zero_optimization": {
            "stage": 3,
            "offload_param": {"device": "cpu", "pin_memory": True},
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "overlap_comm": True,
            "contiguous_gradients": True
        },
        "bf16": {"enabled": True},
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 2,
        "zero_allow_untested_optimizer": True
    }
    with open("ds_config.json", "w") as f:
        json.dump(ds_config, f, indent=2)

    # Step 6: Set up training arguments and the Trainer.
    print("Setting up training arguments and trainer...")
    training_args = TrainingArguments(
        output_dir="./fine_tuned_model",
        overwrite_output_dir=True,
        eval_strategy="epoch",
        learning_rate=2e-4,
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=2,
        num_train_epochs=3,
        weight_decay=0.01,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        report_to="none",
        bf16=True,
        optim="adamw_bnb_8bit",
        deepspeed="./ds_config.json"
    )
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Step 7: Fine-tune the model.
    print("Starting fine-tuning...")
    trainer.train()

    # Save the fine-tuned model and tokenizer.
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
    print("Saving model and tokenizer...")
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print(f"Fine-tuning complete. Model saved to {FINE_TUNED_MODEL_DIR}.")

if __name__ == "__main__":
    main()
