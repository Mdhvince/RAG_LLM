import os
import shutil
from pathlib import Path

import torch
from langchain import PromptTemplate, OpenAI
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory, ConversationTokenBufferMemory, \
    ConversationSummaryBufferMemory
from langchain.retrievers import SelfQueryRetriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings.huggingface import HuggingFaceEmbeddings, HuggingFaceInstructEmbeddings
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM, \
    RagTokenizer, RagTokenForGeneration
from peft import PeftModel, TaskType


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

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
    splits = text_splitter.split_documents(docs)

    vectordb = Chroma.from_documents(  # this takes a while
        documents=splits,
        embedding=embedding,
        persist_directory=persist_directory
    )
    vectordb.persist()

    return vectordb

def get_model(model_name, peft_model_path, task="text2text-generation", gen_config=None, use_peft=True):
    tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")

    if use_peft:
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

        peft_model = PeftModel.from_pretrained(
            model,
            peft_model_path,
            orch_dtype=torch.bfloat16,
            is_trainable=False,
            generation_config=gen_config
        )
        model = peft_model
    else:
        model = model_name

    pipe = pipeline(
        task=task,
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        generation_config=gen_config
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm


chat_history = []


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent
    RELOAD = False
    persist_directory = str(PROJECT_ROOT / "docs/chroma/")
    documents_directory = str(PROJECT_ROOT / "docs")
    max_completion_length = 3000
    embedding = HuggingFaceEmbeddings()

    if RELOAD:
        if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
        vectordb = create_vectors(
            persist_directory,
            embedding,
            doc_dir=documents_directory
        )
    else:
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)
    print("done.")


    gen_config = GenerationConfig(temperature=0, max_length=max_completion_length)

    llm = get_model(
        model_name="google/flan-t5-base",
        peft_model_path=str(PROJECT_ROOT / "ft-alignment/peft-checkpoint-local"),
        task="text2text-generation",
        gen_config=gen_config,
        use_peft=True
    )

    memory = ConversationBufferMemory(
        llm=llm,
        memory_key="chat_history",
        input_key="question",
        output_key="answer",
        return_messages=True
    )


    qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=vectordb.as_retriever(search_type="mmr", search_kwargs={"k": 4}),
        memory=memory,
        verbose=False,
        return_source_documents=True,
    )

    while True:
        query = input("User: ")
        if query == "exit": break

        in_key = {"question": query}
        in_key.update(memory.load_memory_variables({}))
        result = qa(in_key)

        print(f"Bot: {result['answer']}")

        meta = result["source_documents"][0].metadata
        source = str(Path(meta["source"]).name)
        page = meta["page"]
        print(f"Source: {source} -> Page: {page}\n")






