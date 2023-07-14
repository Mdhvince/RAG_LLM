from pathlib import Path

from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM

from langchain_helpers.cogninova_search import CogninovaSearch
from langchain_helpers.retrieval_template import RetrievalTemplate


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    persist_dir = str(PROJECT_ROOT / "docs/chroma/")
    docs_dir = str(PROJECT_ROOT / "docs")
    embedding = HuggingFaceEmbeddings()  # I could have used directly an encoder-based model like "BERT"
    vdb_type = "chroma"

    model_name = "google/flan-t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
    gen_config = GenerationConfig(
        temperature=0.1, max_new_tokens=200, top_k=50, top_p=1.0, eos_token_id=tokenizer.eos_token_id
    )
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    cs = CogninovaSearch(model_name, gen_config, llm, embedding)
    rtp = RetrievalTemplate()

    cs.load_document(docs_dir, persist_dir, chk_size=1500, chk_overlap=500, vdb_type=vdb_type)
    cs.load_vector_database(persist_dir, vdb_type=vdb_type)

    # Let's go
    query = "Who is the author?"
    search_result = cs.search(query, k=5, search_type="similarity", filter_on=None)
    natural_answer = cs.answer(query, search_result, chain_type="refine", rtp=rtp, verbose=False)
    print(natural_answer)














