import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def load_model(model_name="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
    return model, tokenizer


def n_trainable_parameters(model):
    trainable_params = 0
    all_params = 0

    for _, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    return (
        f"""
        Number of trainable parameters: {trainable_params}
        Number of all parameters: {all_params}
        Percentage of trainable parameters: {trainable_params / all_params * 100:.2f}%
        """
    )
