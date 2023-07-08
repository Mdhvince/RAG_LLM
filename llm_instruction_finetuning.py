"""
Instruction Fine-tuning is the most common way of fine-tuning a pretrained model.

> Data annotation (Input and Target): Can be on single or multiple tasks.
> Fine-tuning with cross entropy loss (comparison of the probability distribution of tokens).
> Evaluation using benchmarks.
"""
from datasets import load_dataset

from utils import load_model, n_trainable_parameters

# Dialogue specific functions
def tokenize_dialogue(example, tokenizer):
    """
    Tokenize a dialogue and return the tokenized dialogue.
    """
    start_prompt = "Summarize the following conversation:\n\n"
    end_prompt = "\n\nSummary: "
    prompt = [start_prompt + dialogue + end_prompt for dialogue in example["dialogue"]]
    example["input_ids"] = tokenizer(prompt, padding="max_length", truncation=True, return_tensors="pt").input_ids
    example["labels"] = tokenizer(
        example["summary"], padding="max_length", truncation=True, return_tensors="pt").input_ids

    return example




if __name__ == '__main__':

    hf_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(hf_dataset_name)
    original_model, tokenizer = load_model(model_name="google/flan-t5-base")

    print(n_trainable_parameters(original_model))

    # Preprocessing steps

    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda example: tokenize_dialogue(example, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["id", "topic", "dialogue", "summary"])

    # Subsample the dataset to save time
    tokenized_dataset = tokenized_dataset.filter(lambda example, idx: idx % 100 == 0, with_indices=True)

    # Print some stats
    print("Shape of the dataset:")
    print(f"Train: {tokenized_dataset['train'].shape}")
    print(f"Validation: {tokenized_dataset['validation'].shape}")
    print(f"Test: {tokenized_dataset['test'].shape}\n")
    print(tokenized_dataset)



