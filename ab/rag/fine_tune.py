import os
# Disable wandb logging completely
os.environ["WANDB_MODE"] = "disabled"
os.environ["HF_HUB_DOWNLOAD_TIMEOUT"] = "300"

import json
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
from datasets import Dataset, load_dataset
from ab.nn.api import data  # Import the LEMUR data API

# ---------------------------------------------------------------------------
# Step 1: Prepare Fine-Tuning Data Using the LEMUR API
# ---------------------------------------------------------------------------
# Call the API to get a DataFrame with NN performance and code details.
df = data(only_best_accuracy=False)

def create_example(row):
    # Build a prompt with key context information and expected NN code as the response.
    prompt = (
        f"Task: {row.get('task', 'N/A')}\n"
        f"Dataset: {row.get('dataset', 'N/A')}\n"
        f"Metric: {row.get('metric', 'N/A')}\n"
        f"Epoch: {row.get('epoch', 'N/A')}\n"
        f"Hyperparameters: {row.get('prm', {})}\n"
        f"Dataset Description: [Include Internet-sourced dataset details here]\n"
        "Provide the corresponding NN model code and predicted accuracy."
    )
    response = row.get('nn_code', 'No NN code available.')
    return {"prompt": prompt, "response": response}

# Convert each row of the DataFrame into a fine-tuning example
examples = [create_example(row) for _, row in df.iterrows()]

# Split data into train and validation sets (80/20 split)
split_idx = int(0.8 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# Optionally, save these examples to JSON files for inspection.
os.makedirs("data", exist_ok=True)
with open("data/lemur_train.json", "w", encoding="utf-8") as f:
    json.dump(train_examples, f, indent=2)
with open("data/lemur_val.json", "w", encoding="utf-8") as f:
    json.dump(val_examples, f, indent=2)

# Create Hugging Face datasets from the examples.
train_dataset = Dataset.from_dict({
    "prompt": [ex["prompt"] for ex in train_examples],
    "response": [ex["response"] for ex in train_examples]
})
val_dataset = Dataset.from_dict({
    "prompt": [ex["prompt"] for ex in val_examples],
    "response": [ex["response"] for ex in val_examples]
})

def preprocess_function(examples):
    # Combine prompt and response for causal language modeling.
    combined = [f"{p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    return tokenizer(combined, truncation=True, padding="max_length", max_length=512)

# ---------------------------------------------------------------------------
# Step 2: Load the Distilled Model and Tokenizer
# ---------------------------------------------------------------------------
MODEL_ID = "deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    torch_dtype="auto"
)

# ---------------------------------------------------------------------------
# Step 3: Set Up QLoRA Fine-Tuning Configuration (PEFT)
# ---------------------------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none"
)
model = get_peft_model(base_model, lora_config)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---------------------------------------------------------------------------
# Step 4: Tokenize the Datasets
# ---------------------------------------------------------------------------
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------------------------------------------------------
# Step 5: Set Up Training Arguments and Trainer
# ---------------------------------------------------------------------------
training_args = TrainingArguments(
    output_dir="./fine_tuned_model",
    overwrite_output_dir=True,
    evaluation_strategy="epoch",
    learning_rate=2e-4,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_steps=10,
    save_strategy="epoch",
    save_total_limit=2,
    report_to="none"  # Disable logging to W&B and other services
)

from transformers import Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    data_collator=data_collator,
)

print("Starting fine-tuning...")
trainer.train()

# Save the fine-tuned model and tokenizer for later integration
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuning complete. Model saved to './fine_tuned_model'.")
