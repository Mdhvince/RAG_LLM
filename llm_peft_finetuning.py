"""
Parameter Efficient Fine-tuning

> Data annotation (Input and Target): Can be on single or multiple tasks.
> Fine-tuning with cross entropy loss (comparison of the probability distribution of tokens)

  - SELECTIVE METHOD: Update a subset of the parameters of the model by freezing most of the existing model weights.

  - ADDITIVE METHOD:
    Update the existing model weights by freezing them all and add additional parameters to it. Another Additive method
    is Prompt tuning (different from Prompt Engineering): We add additional trainable tokens (called soft prompt) then
    do supervised learning on them (perform very well for LLMs with +10B parameters).

  - RE-PARAMETERIZATION METHOD (i.e. LoRA - Low Rank Adaptation):
    Reduce the number of parameters to train by creating new low rank transformations of the original model weights.

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
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, PeftConfig

from llm_instruction_finetuning import preprocess_dialogue, evaluation_q
from utils import load_model, n_trainable_parameters



if __name__ == "__main__":
    hf_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(hf_dataset_name)
    original_model, tokenizer = load_model(model_name="google/flan-t5-base")
    tokenized_dataset = preprocess_dialogue(dataset, tokenizer)

    lora_config = LoraConfig(
        r=32,  # Rank of the low rank transformation
        lora_alpha=32,  # Number of low rank transformations
        target_modules=["q", "v"],  # Modules to apply the low rank transformation (Query and Value of the attention)
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,  # FLAN-T5
    )

    # Add LoRA adapter layers/parameters to the model
    peft_model = get_peft_model(original_model, lora_config)
    print(f"Number of trainable parameters: {n_trainable_parameters(peft_model)}")

    output_dir = Path(f"peft_dialogue_summary_training_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    output_dir.mkdir(exist_ok=True)

    TRAIN = False
    EVALUATE = True

    if TRAIN:
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            auto_find_batch_size=True,
            learning_rate=1e-3,
            num_train_epochs=1,
            logging_steps=1,
            max_steps=1
        )
        trainer = Trainer(
            model=peft_model,
            args=training_args,
            train_dataset=tokenized_dataset["train"]
        )
        trainer.train()

        model_path = Path("peft-checkpoint-local")
        model_path.mkdir(exist_ok=True)
        trainer.model.save_pretrained(model_path)
        tokenizer.save_pretrained(model_path)

    if EVALUATE:
        gen_config = GenerationConfig(
            do_sample=True, max_new_tokens=200, top_k=50, top_p=0.95, num_beams=1
        )

        peft_model = PeftModel.from_pretrained(
            original_model,
            "peft-checkpoint-local",
            torch_dtype=torch.bfloat16,
            is_trainable=False
        )
        print(f"Number of trainable parameters: {n_trainable_parameters(peft_model)}")  # should be 0

        dialogues = dataset["test"][0:10]["dialogue"]
        human_baseline_summaries = dataset["test"][0:10]["summary"]

        df_human, df_rouge, original_model_results, peft_model_results = evaluation_q(
            original_model, peft_model, dialogues, human_baseline_summaries, tokenizer, gen_config
        )

        # Percentage improvement of the instruction fine-tuned model over the original model
        improvement = (
                np.array(list(peft_model_results.values())) - np.array(list(original_model_results.values()))
        )
        for key, value in zip(peft_model_results.keys(), improvement):
            print(f"{key}: {value * 100:.2f} %")















