from pathlib import Path

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM

from langchain_helpers.cogninova_search import CogninovaSearch
from langchain_helpers.retrieval_template import RetrievalTemplate


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    RELOAD = False
    persist_dir = str(PROJECT_ROOT / "docs/chroma/")
    documents_directory = str(PROJECT_ROOT / "docs")
    embedding = HuggingFaceEmbeddings()
    vdb_type = "chroma"

    model_name = "google/flan-t5-small"
    gen_config = GenerationConfig(
        temperature=0.1, max_new_tokens=200, top_k=50, top_p=1.0
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")

    cs = CogninovaSearch(model_name, gen_config, llm)
    rtp = RetrievalTemplate()

    if RELOAD:
        loaded_docs = CogninovaSearch.load_document(documents_directory)
        cs.store_embeddings(embedding, persist_dir, loaded_docs, 1500, 500, vector_db_type=vdb_type)

    vectordb = cs.vector_database(embedding, persist_dir, vector_db_type=vdb_type)

    # Let's go
    query = "Who is the author?"
    search_result = cs.search(query, k=5, search_type="mmr", filter_on=None)
    natural_answer = cs.answer(query, search_result, chain_type="refine", rtp=rtp, verbose=False)
    print(natural_answer)














