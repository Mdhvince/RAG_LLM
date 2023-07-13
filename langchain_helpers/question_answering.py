import os

import openai
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.vectorstores import DocArrayInMemorySearch, Chroma
from langchain.indexes import VectorstoreIndexCreator
from langchain.embeddings import OpenAIEmbeddings


if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0)
    embeddings = OpenAIEmbeddings()

    loader = CSVLoader(file_path="my_data.csv")
    ###################################################################################################################

    index_creator = VectorstoreIndexCreator(
        vector_cls=DocArrayInMemorySearch,  # will be stored in memory here
        embedding=embeddings,
    )
    index = index_creator.from_loaders([loader])  # allows to search the most relevant embeddings for a given query

    # Using the Chroma database
    # persist_directory = "path/to/my/db"
    # index_creator = VectorstoreIndexCreator(
    #     vector_cls=Chroma,  # will be stored in Chroma db
    #     embedding=embeddings,
    #     vectorstore_kwargs = {"persist_directory": persist_directory}
    # )
    # index = index_creator.from_loaders([loader])
    # index.vectorstore.persist()

    # To load only (previously created)
    # index = VectorstoreIndexCreator().from_persistent_index(path=persist_directory)



    query = "What are the top 5 plants with best profit in 2023 ?"  # based on the data in my_data.csv (in vectorstore)
    response = index.query(question=query, llm=llm)
    print(response)

    ###################################################################################################################
    # Same as above, but step by step
    docs = loader.load()


    db = DocArrayInMemorySearch.from_documents(documents=docs, embedding=embeddings)
    retriever = db.as_retriever()
    chain_type = "refine"  # "stuff, map_reduce (good for summary), refine, map_rerank"
    qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, verbose=True)

    query = "What are the top 5 plants with best profit in 2023 ?"
    response = qa.run(query=query)


