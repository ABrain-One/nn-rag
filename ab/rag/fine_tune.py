import os
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
# You can add filters as needed; here we retrieve all data.
df = data(only_best_accuracy=False)

# For demonstration, we'll create fine-tuning examples where:
# - The prompt is constructed from key context fields.
# - The response is the neural network code (or a key performance metric).

def create_example(row):
    # Build the prompt from selected columns.
    # For example, combine task, dataset, metric, epoch, and hyperparameters.
    prompt = (
        f"Task: {row.get('task', 'N/A')}\n"
        f"Dataset: {row.get('dataset', 'N/A')}\n"
        f"Metric: {row.get('metric', 'N/A')}\n"
        f"Epoch: {row.get('epoch', 'N/A')}\n"
        f"Hyperparameters: {row.get('prm', {})}\n"
        f"Dataset Description: [Add Internet-sourced dataset details if available]\n"
        "Provide the corresponding NN model code and predicted accuracy."
    )
    # Use the neural network code as the response.
    # Alternatively, you could use accuracy or a combination.
    response = row.get('nn_code', 'No NN code available.')
    return {"prompt": prompt, "response": response}

# Apply the conversion to all rows in the DataFrame
examples = [create_example(row) for idx, row in df.iterrows()]

# Split data into train and validation sets (e.g., 80/20 split)
split_idx = int(0.8 * len(examples))
train_examples = examples[:split_idx]
val_examples = examples[split_idx:]

# Save these examples to JSON files (optional, for inspection)
os.makedirs("data", exist_ok=True)
with open("data/lemur_train.json", "w", encoding="utf-8") as f:
    json.dump(train_examples, f, indent=2)
with open("data/lemur_val.json", "w", encoding="utf-8") as f:
    json.dump(val_examples, f, indent=2)

# Alternatively, create a Hugging Face Dataset directly:
train_dataset = Dataset.from_dict({"prompt": [ex["prompt"] for ex in train_examples],
                                   "response": [ex["response"] for ex in train_examples]})
val_dataset = Dataset.from_dict({"prompt": [ex["prompt"] for ex in val_examples],
                                 "response": [ex["response"] for ex in val_examples]})

# For causal language modeling, combine prompt and response as the input text.
def preprocess_function(examples):
    combined = [f"{p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    # Tokenize with truncation/padding; adjust max_length as needed.
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

# Set up QLoRA configuration using PEFT
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],  # Adjust based on model architecture
    lora_dropout=0.05,
    bias="none"
)

# Wrap the base model with QLoRA to get the fine-tuning model
model = get_peft_model(base_model, lora_config)
print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

# ---------------------------------------------------------------------------
# Step 3: Tokenize the Datasets
# ---------------------------------------------------------------------------
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=["prompt", "response"])

from transformers import DataCollatorForLanguageModeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# ---------------------------------------------------------------------------
# Step 4: Set Up Training Arguments and Fine-Tune
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

# Save the fine-tuned model and tokenizer
model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")
print("Fine-tuning complete. Model saved to './fine_tuned_model'.")
