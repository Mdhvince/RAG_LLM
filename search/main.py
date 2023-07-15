from pathlib import Path

from langchain import PromptTemplate
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig

from search.cogninova_memory import CogninovaMemory
from search.cogninova_search import CogninovaSearch
from search.cogninova_template import CogninovaTemplate


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    persist_dir = str(PROJECT_ROOT / "docs/chroma/")
    docs_dir = str(PROJECT_ROOT / "docs")
    embedding = HuggingFaceEmbeddings()
    vdb_type = "chroma"
    chain_type = "refine"

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    gen_config = GenerationConfig(
        temperature=0.1, max_new_tokens=200, top_k=50, top_p=1.0
    )
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    cm = CogninovaMemory()
    ct = CogninovaTemplate()
    cs = CogninovaSearch(model_name, gen_config, llm, embedding)

    cs.load_document(docs_dir, persist_dir, chk_size=1500, chk_overlap=500, vdb_type=vdb_type)
    cs.load_vector_database(persist_dir, vdb_type=vdb_type)

    # Let's go chat with the knowledge base
    query = "Test"
    cm.update(query)

    if cm.is_full():
        # Here I generate a new query (the standalone question)
        chat_history: str = cm.get_chat_history()
        p_template = PromptTemplate(template=ct.standalone_question_template, input_variables=["chat_history"])
        prompt = p_template.format(chat_history=chat_history)
        query = cs.run_inference(prompt)
        cm.optimize(query)

    search_result = cs.search(query, k=5, search_type="similarity", filter_on=None)
    natural_answer = cs.answer(query, search_result, template_obj=ct, chain_type=chain_type)
    cm.update(natural_answer)

    print(f"\nBot: {natural_answer}")














