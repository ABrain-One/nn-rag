"""
Fine-Tuning Pipeline for DeepSeek-R1-Distill-Qwen-1.5B

Key improvements:
- Incorporates real GitHub or dataset snippets into training examples.
- Sometimes includes a "refactor" style prompt for advanced code modifications.
"""

import os
import json
import random
import torch

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
from utils.data_loader import load_full_corpus
from utils.retrieval import CodeRetrieval
from config.config import (
    DATASET_DESC_DIR, GITHUB_REPO_DIR, CODE_EMBEDDING_MODEL_NAME,
    EMBEDDING_BATCH_SIZE, FAISS_INDEX_PATH, TOP_K_RETRIEVAL, 
    FINE_TUNED_MODEL_DIR, HF_TOKEN
)

MAX_LENGTH = 4096  # Maximum tokens for training examples

# -------------------------------------------------------
# 1. Create a retrieval system for training-time snippet injection
# -------------------------------------------------------
def retrieve_best_snippet(dataset_name, retrieval_system, fallback_query="pytorch"):
    """
    Uses the retrieval system to get the top snippet relevant to the dataset_name.
    If none found or dataset_name is 'N/A', does a fallback search on 'pytorch'
    (or any fallback query you want).
    """
    if not dataset_name or dataset_name == "N/A":
        dataset_name = fallback_query  # e.g. searching for "pytorch" or "cnn"

    results = retrieval_system.search(dataset_name, top_k=3)  # get top 3
    if not results:
        # If no results, do fallback
        results = retrieval_system.search(fallback_query, top_k=3)
        if not results:
            return "No external snippet found."
    
    # Just pick the first of the top results (distance-based)
    best = results[0]
    snippet_text = best["text"]
    return snippet_text

# -------------------------------------------------------
# 2. Create training examples
# -------------------------------------------------------
def create_example(row, retrieval_system):
    """
    Converts a row from the LEMUR dataset into a training example.
    We retrieve a snippet from the corpus via CodeRetrieval,
    then 30% of the time do a "refactor" style prompt.
    """
    task = row.get('task', 'N/A')
    dataset_name = row.get('dataset', 'N/A')
    metric = row.get('metric', 'N/A')
    hyperparams = row.get('prm', {})
    nn_code = row.get('nn_code', 'No NN code available.')
    accuracy = row.get('accuracy', 'N/A')
    epoch = row.get('epoch', 'N/A')

    # 2.1: Retrieve snippet from the index
    external_snippet = retrieve_best_snippet(dataset_name, retrieval_system)

    # 2.2: Decide if we do "refactor" or direct
    do_refactor = (random.random() < 0.3)  # 30% chance

    if do_refactor:
        prompt = (
            f"Task: {task}\n"
            f"Dataset: {dataset_name}\n"
            f"Metric: {metric}\n"
            f"Hyperparameters: {hyperparams}\n"
            f"External Snippet:\n{external_snippet}\n\n"
            "We have an existing model below. Please refactor or adapt the model code to match the above snippet or dataset, "
            "then provide predicted accuracy and epoch.\n"
            f"Existing Model Code:\n{nn_code}\n"
            "Refactor it now."
        )
        response = (
            "Refactored NN Code:\n"
            f"{nn_code}\n"
            f"Predicted Accuracy: {accuracy}\n"
            f"Epoch: {epoch}"
        )
    else:
        prompt = (
            f"Task: {task}\n"
            f"Dataset: {dataset_name}\n"
            f"Metric: {metric}\n"
            f"Hyperparameters: {hyperparams}\n"
            f"External Snippet:\n{external_snippet}\n\n"
            "Based on all the information above, provide a corresponding NN model code along with predicted accuracy and epoch."
        )
        response = (
            f"NN Code:\n{nn_code}\n"
            f"Predicted Accuracy: {accuracy}\n"
            f"Epoch: {epoch}"
        )

    return {"prompt": prompt, "response": response}

def preprocess_function(examples, tokenizer):
    """
    Combines prompt and response into a single text and tokenizes it.
    """
    combined = [f"{p}\nResponse: {r}" for p, r in zip(examples["prompt"], examples["response"])]
    tokenized = tokenizer(
        combined, 
        truncation=True, 
        padding="max_length", 
        max_length=MAX_LENGTH
    )
    tokenized["labels"] = tokenized["input_ids"].copy()
    return tokenized

# -------------------------------------------------------
# 3. Main Fine-Tuning Function
# -------------------------------------------------------
def main():
    # Check if a fine-tuned model already exists; if so, skip training to save resources.
    if os.path.isdir(FINE_TUNED_MODEL_DIR):
        print(f"Fine-tuned model already exists at {FINE_TUNED_MODEL_DIR}. Skipping training.")
        return

    # 3.1: Ensure data is set up
    print("Running setup to ensure external data sources are available...")
    run_setup()

    # 3.2: Load the corpus for retrieval
    print("Loading corpus data for retrieval (GitHub + dataset desc) ...")
    corpus_data = load_full_corpus(DATASET_DESC_DIR, GITHUB_REPO_DIR)
    if not corpus_data:
        raise ValueError("Corpus is empty; cannot proceed with snippet retrieval for training.")
    
    # 3.3: Build or load the FAISS index from the corpus
    print("Building FAISS index for training-time retrieval ...")
    retrieval_system = CodeRetrieval(
        model_name=CODE_EMBEDDING_MODEL_NAME,
        batch_size=EMBEDDING_BATCH_SIZE,
        index_path=FAISS_INDEX_PATH
    )
    # If you already have an index, you can do retrieval_system.load_index(...), 
    # but let's always build it here for fresh training
    retrieval_system.build_index(corpus_data)

    # 3.4: Prepare fine-tuning data from LEMUR
    print("Fetching LEMUR data for fine-tuning (only_best_accuracy=False)...")
    df = data(only_best_accuracy=False).reset_index(drop=True)

    # 3.5: Convert each row to a training example with snippet retrieval
    all_examples = []
    for _, row in df.iterrows():
        ex = create_example(row, retrieval_system)
        all_examples.append(ex)

    random.shuffle(all_examples)
    split_idx = int(0.8 * len(all_examples))
    train_examples = all_examples[:split_idx]
    val_examples = all_examples[split_idx:]

    # 3.6: Save for inspection
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

    # 3.7: Load the base model + tokenizer
    MODEL_ID_local = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    print(f"Loading base model/tokenizer from {MODEL_ID_local}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID_local, trust_remote_code=True, use_auth_token=HF_TOKEN)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID_local,
        trust_remote_code=True,
        torch_dtype="auto",
        use_auth_token=HF_TOKEN
    )

    # 3.8: Setup QLoRA (PEFT)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none"
    )
    model = get_peft_model(base_model, lora_config)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    print("Trainable parameters:", sum(p.numel() for p in model.parameters() if p.requires_grad))

    # 3.9: Tokenize data
    print("Tokenizing data...")
    tokenized_train = train_dataset.map(
        lambda e: preprocess_function(e, tokenizer), 
        batched=True, 
        remove_columns=["prompt", "response"]
    )
    tokenized_val = val_dataset.map(
        lambda e: preprocess_function(e, tokenizer), 
        batched=True, 
        remove_columns=["prompt", "response"]
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # 3.10: DeepSpeed config
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

    # 3.11: Training args
    training_args = TrainingArguments(
        output_dir=FINE_TUNED_MODEL_DIR,
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

    print("Starting fine-tuning...")
    trainer.train()

    # 3.12: Save final model
    os.makedirs(FINE_TUNED_MODEL_DIR, exist_ok=True)
    print("Saving model and tokenizer...")
    model.save_pretrained(FINE_TUNED_MODEL_DIR)
    tokenizer.save_pretrained(FINE_TUNED_MODEL_DIR)
    print(f"Fine-tuning complete. Model saved to {FINE_TUNED_MODEL_DIR}.")

if __name__ == "__main__":
    main()