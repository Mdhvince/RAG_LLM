from pathlib import Path

import torch
from peft import PeftModel
from transformers import GenerationConfig, AutoModelForSeq2SeqLM, AutoTokenizer, AutoModelForCausalLM

from utils import load_model, n_trainable_parameters

if __name__ == '__main__':

    org_model, tokenizer = load_model(model_name="google/flan-t5-base")
    model_path = Path("peft-checkpoint-local")
    gen_config = GenerationConfig(do_sample=True, max_new_tokens=200, top_k=50, top_p=0.95, num_beams=1)

    peft_model = PeftModel.from_pretrained(org_model, model_path, torch_dtype=torch.bfloat16, is_trainable=False)

    print(n_trainable_parameters(peft_model))

    while True:
        # Get the prompt from user input
        prompt = input("User: ")

        if prompt == "exit":
            break

        input_ids = tokenizer(prompt, return_tensors="pt").input_ids

        model_output = peft_model.generate(input_ids=input_ids, generation_config=gen_config)
        text = tokenizer.decode(model_output[0], skip_special_tokens=True)

        print(f"Bot: {text}")
        print()




