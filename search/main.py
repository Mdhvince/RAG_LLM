from langchain import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoModelForSeq2SeqLM

import utils
from search.cogninova_memory import CogninovaMemory
from search.cogninova_search import CogninovaSearch
from search.cogninova_template import CogninovaTemplate


if __name__ == "__main__":
    cfg, cfg_vs, cfg_search, cfg_model = utils.get_config()

    llm = AutoModelForSeq2SeqLM.from_pretrained(cfg_model.get("name"), device_map="auto")
    embedding = HuggingFaceEmbeddings()

    cm = CogninovaMemory()
    ct = CogninovaTemplate()
    cs = CogninovaSearch(cfg_model, cfg_search, cfg_vs, llm, embedding)


    while True:
        # Let's go chat with the knowledge base
        query = input("You: ")
        if query == "exit":
            break

        cm.update(query)

        if cm.is_full():
            # Here I generate a new query (the standalone question)
            chat_history: str = cm.get_chat_history()
            p_template = PromptTemplate(template=ct.standalone_question_template, input_variables=["chat_history"])
            prompt = p_template.format(chat_history=chat_history)
            query = cs.run_inference(prompt)
            cm.optimize(query)

        search_result = cs.search(query, filter_on=None)
        natural_answer = cs.answer(query, search_result, template_obj=ct)
        cm.update(natural_answer)

        print(f"\nBot: {natural_answer}")














