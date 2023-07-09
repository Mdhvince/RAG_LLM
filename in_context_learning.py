from datasets import load_dataset
from transformers import GenerationConfig

from utils import load_model


def make_prompt(dataset, example_indices, index_to_summarize):
    prompt = ""
    for index in example_indices:
        dialogue = dataset["test"][index]["dialogue"]
        summary = dataset["test"][index]["summary"]

        # prompt with target summary
        prompt += f"""
        Summarize the following conversation:
        {dialogue}

        Summary:
        {summary} \n\n\n
        """

    # Add final example to summarize (without target summary)
    dialogue = dataset["test"][index_to_summarize]["dialogue"]
    prompt += f"""
    Summarize the following conversation:
    {dialogue}

    Summary:
    """
    return prompt


if __name__ == '__main__':

    hf_dataset_name = "knkarthick/dialogsum"
    dataset = load_dataset(hf_dataset_name)
    model, tokenizer = load_model()
    gen_config = GenerationConfig(do_sample=True, max_new_tokens=100, top_k=50, top_p=0.95, temperature=1.1)


    dash_line = "-" * 100
    ast_line = "*" * 100

    ICL = "one"  # "zero", "one" or "few" shot


    if ICL == "zero":
        print(f"\n\n{ast_line} \nPROMPT ENGINEERING: 0 shot")

        example_indices = [20, 40]

        for n, index in enumerate(example_indices):
            dialogue = dataset["test"][index]["dialogue"]
            summary = dataset["test"][index]["summary"]

            prompt = f"""
                Summarize the following conversation:
                {dialogue}
    
                Summary:
            """

            input_ids = tokenizer(prompt, return_tensors="pt").input_ids
            outputs = model.generate(input_ids, generation_config=gen_config)[0]
            readable_output = tokenizer.decode(outputs, skip_special_tokens=True)

            print(f"{dash_line} \nExample {n}")
            print(summary)
            print(f"{dash_line} \nMODEL GENERATED SUMMARY")
            print(readable_output)

    else:
        print(f"\n\n{ast_line} \nPROMPT ENGINEERING: 1 shot")

        example_indices = [40]  # for few shot, use add more indices in this list
        index_to_summarize = 20
        one_shot_prompt = make_prompt(dataset, example_indices, index_to_summarize)

        summary = dataset["test"][index_to_summarize]["summary"]

        input_ids = tokenizer(one_shot_prompt, return_tensors="pt").input_ids
        outputs = model.generate(input_ids, generation_config=gen_config)[0]
        readable_output = tokenizer.decode(outputs, skip_special_tokens=True)

        print(f"{dash_line} \nBASELINE HUMAN SUMMARY")
        print(summary)
        print(f"{dash_line} \nMODEL GENERATED SUMMARY")
        print(readable_output)






