import os
import shutil
from pathlib import Path

import torch
from langchain import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM

from langchain_helpers.retrieval_template import RetrievalTemplate


def create_vectors(persist_directory, embedding, doc_dir="docs"):
    """
    Load documents from a directory
    """
    docs = []
    # PDF only for the moment
    file_dir = Path(doc_dir)

    for file in file_dir.iterdir():
        if file.suffix == ".pdf":
            loader = PyPDFLoader(str(file))
            docs.extend(loader.load())

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=500)
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(  # this takes a while
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()

    return vectordb


def answer(llm, prompt, gen_config, tokenizer):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    model_output = llm.generate(input_ids=input_ids, generation_config=gen_config)
    text = tokenizer.decode(model_output[0], skip_special_tokens=True)
    return text


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    RELOAD = False
    persist_directory = str(PROJECT_ROOT / "docs/chroma/")
    documents_directory = str(PROJECT_ROOT / "docs")
    embedding = HuggingFaceEmbeddings()

    model = "google/flan-t5-small"
    task = "text2text-generation"
    gen_config = GenerationConfig(
        temperature=0.1, max_new_tokens=50, top_k=50, top_p=1.0,
    )
    tokenizer = AutoTokenizer.from_pretrained(model, device_map="auto")
    llm = AutoModelForSeq2SeqLM.from_pretrained(model, device_map="auto")

    if RELOAD:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        vectordb = create_vectors(
            persist_directory,
            embedding,
            doc_dir=documents_directory
        )
    else:
        vectordb = Chroma(
            persist_directory=persist_directory, embedding_function=embedding)
    ##################################################################################################################

    rtp = RetrievalTemplate()

    query = "What is AWS Lambda ?"
    search_type = "mmr"
    chain_type = "refine"
    k = 5  # number of relevant documents chunk to return. If filter is set on a document, return k chunk in the doc
    filter_on = None  # filter the documents to search in ({"source":"docs/cs229_lectures/Lecture03.pdf"})


    assert search_type in ["similarity", "mmr"], f"search_type must in ['similarity', 'mmr'] got {search_type}"

    if search_type == "similarity":
        result = vectordb.similarity_search(query, k=k, filter=filter_on)
    else:  # mmr
        result = vectordb.max_marginal_relevance_search(query, k=k, filter=filter_on)


    if chain_type == "stuff":
        document_separator = "\n<<<<.>>>>\n"
        context = []
        for res in result:
            chunked_content = res.page_content
            context.append(res.page_content)

        context_str = document_separator.join(context)

        prompt_template = PromptTemplate(template=rtp.stuff_template, input_variables=["context", "question"])
        prompt = prompt_template.format(context=context_str, question=query)
        guess = answer(llm, prompt, gen_config, tokenizer)
        print(guess)

    elif chain_type == "refine":
        # First guess
        first_context = result[0].page_content
        prompt_template = PromptTemplate(template=rtp.refine_template_start, input_variables=["context", "question"])
        prompt = prompt_template.format(context=first_context, question=query)

        guess = answer(llm, prompt, gen_config, tokenizer)
        print(f"Guess: {guess}")

        # refine the answer
        other_contexts = result[1:]

        if len(other_contexts) > 0:
            for next_context in other_contexts:
                next_context = next_context.page_content
                inputs = ["question", "guess", "context"]
                prompt_template = PromptTemplate(template=rtp.refine_template_next, input_variables=inputs)
                prompt = prompt_template.format(context=next_context, question=query, guess=guess)
                guess = answer(llm, prompt, gen_config, tokenizer)
                print(f"Guess: {guess}")

        print(guess)





