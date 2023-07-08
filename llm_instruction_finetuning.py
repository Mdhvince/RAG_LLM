"""
Instruction Fine-tuning is the most common way of fine-tuning a pretrained model.

> Data annotation (Input and Target): Can be on single or multiple tasks.
> Fine-tuning with cross entropy loss (comparison of the probability distribution of tokens).
> Evaluation using benchmarks.
"""
import datetime
from pathlib import Path

import torch
import evaluate
import pandas as pd
import numpy as np
from datasets import load_dataset
from transformers import TrainingArguments, Trainer, AutoModelForSeq2SeqLM, GenerationConfig

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

def preprocess_dialogue(dataset, tokenizer, print_stats=True):
    """
    Preprocess the dataset by tokenizing the input and target.
    """
    # Tokenize the dataset
    tokenized_dataset = dataset.map(lambda example: tokenize_dialogue(example, tokenizer), batched=True)
    tokenized_dataset = tokenized_dataset.remove_columns(["id", "topic", "dialogue", "summary"])

    # Subsample the dataset to save time
    tokenized_dataset = tokenized_dataset.filter(lambda example, idx: idx % 100 == 0, with_indices=True)

    if print_stats:
        print("Shape of the dataset:")
        print(f"Train: {tokenized_dataset['train'].shape}")
        print(f"Validation: {tokenized_dataset['validation'].shape}")
        print(f"Test: {tokenized_dataset['test'].shape}\n")
        print(tokenized_dataset)

    return tokenized_dataset


def evaluation_q(original_model, tuned_model, dialogues, human_baseline_summaries, tokenizer, gen_config):
    """
    Qualitative (human) and Quantitative (Rouge) evaluation of the summaries generated by the original and tuned model.
    :param original_model: Original model
    :param tuned_model: Fine-Tuned model
    :param dialogues: Dialogues to be summarized
    :param human_baseline_summaries: Human generated summaries
    :param tokenizer: Tokenizer
    :param gen_config: Generation config
    :return:
    """

    original_model_summaries = []
    tuned_model_summaries = []

    for _, dialogue in enumerate(dialogues):
        prompt = f"Summarize the following conversation:\n\n{dialogue}\n\nSummary: "
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        original_model_output = original_model.generate(input_ids=input_ids, generation_config=gen_config)
        original_model_text = tokenizer.decode(original_model_output[0], skip_special_tokens=True)
        original_model_summaries.append(original_model_text)

        tuned_model_output = tuned_model.generate(input_ids=input_ids, generation_config=gen_config)
        instruct_model_text = tokenizer.decode(tuned_model_output[0], skip_special_tokens=True)
        tuned_model_summaries.append(instruct_model_text)

    # DF for qualitative evaluation (human evaluation)
    zipped_sum = list(zip(human_baseline_summaries, original_model_summaries, tuned_model_summaries))
    df_human = pd.DataFrame(zipped_sum, columns=["Human Baseline", "Original Model", "Instruction Fine-tuned Model"])

    # DF for quantitative evaluation (ROUGE scores)
    rouge = evaluate.load("rouge")
    original_model_results = rouge.compute(
        predictions=original_model_summaries,
        references=human_baseline_summaries[0:len(original_model_summaries)],
        use_stemmer=True
    )
    tuned_model_results = rouge.compute(
        predictions=tuned_model_summaries,
        references=human_baseline_summaries[0:len(tuned_model_summaries)],
        use_stemmer=True
    )
    zipped_rouge = list(zip(original_model_results, tuned_model_results))
    df_rouge = pd.DataFrame(zipped_rouge, columns=["Original Model", "Instruction Fine-tuned Model"])

    return df_human, df_rouge, original_model_results, tuned_model_results


if __name__ == '__main__':
    output_dir = Path(f"dialogue_summary_training_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(exist_ok=True)

    TRAIN = False
    EVALUATE = False

    hf_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(hf_dataset_name)
    original_model, tokenizer = load_model(model_name="google/flan-t5-base")

    # print(n_trainable_parameters(original_model))
    tokenized_dataset = preprocess_dialogue(dataset, tokenizer)

    if TRAIN:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            learning_rate=1e-5,
            num_train_epochs=1,
            weight_decay=0.01,
            logging_steps=1,
            max_steps=1
        )
        trainer = Trainer(
            model=original_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"],
            eval_dataset=tokenized_dataset["validation"]
        )

        trainer.train()  # Fine-tune the model
        model_path = Path("full-ft-checkpoint-local")
        model_path.mkdir(exist_ok=True)
        trainer.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)


    if EVALUATE:
        gen_config = GenerationConfig(
            do_sample=True, max_new_tokens=200, top_k=50, top_p=0.95, num_beams=1
        )

        instruct_model = AutoModelForSeq2SeqLM.from_pretrained("", torch_dtype=torch.bfloat16)
        instruct_model.eval()

        dialogues = dataset["test"][0:10]["dialogue"]
        human_baseline_summaries = dataset["test"][0:10]["summary"]

        df_human, df_rouge, original_model_results, instruct_model_results = evaluation_q(
            original_model, instruct_model, dialogues, human_baseline_summaries, tokenizer, gen_config
        )

        # Percentage improvement of the instruction fine-tuned model over the original model
        improvement = (
                np.array(list(instruct_model_results.values())) - np.array(list(original_model_results.values()))
        )
        for key, value in zip(instruct_model_results.keys(), improvement):
            print(f"{key}: {value * 100:.2f}")






