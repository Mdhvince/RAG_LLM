import os
import re
import shutil
from pathlib import Path

from langchain import PromptTemplate
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from transformers import AutoTokenizer


class CogninovaSearch:
    def __init__(self, model_name, generation_config, llm, embedding):
        self.gen_config = generation_config
        self.llm = llm
        self.embedding = embedding
        self.vector_db = None
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")


    def load_document(self, document_dir, persist_dir, chk_size=1500, chk_overlap=500, vdb_type="chroma"):
        loaded_docs = []
        if isinstance(document_dir, str):
            document_dir = Path(document_dir)

        loaded_docs_dir = document_dir / ".loaded_docs/"
        loaded_docs_dir.mkdir(exist_ok=True)
        count_new_files_loaded = 0
        for file in document_dir.iterdir():
            is_new_file = not (loaded_docs_dir / file.name).exists()

            if not is_new_file: print(f"Skipping {file.name} since it is already loaded")

            if file.suffix == ".pdf" and is_new_file:
                print(f"Loading {file.name}")
                loader = PyPDFLoader(str(file))
                data_txt = loader.load()
                loaded_docs.extend(data_txt)

                shutil.copy(str(file), str(loaded_docs_dir / file.name))  # Copy the file to the loaded_docs_dir
                count_new_files_loaded += 1

        if count_new_files_loaded > 0:
            print(f"Loaded {count_new_files_loaded} new files. Creating embeddings...")
            self._store_embeddings(persist_dir, loaded_docs, chk_size, chk_overlap, vdb_type)
            print(f"Created embeddings for {count_new_files_loaded} new files.")

    def _store_embeddings(self, persist_dir, loaded_docs, chk_size, chk_overlap, vdb_type):

        text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            self.tokenizer, chunk_size=chk_size, chunk_overlap=chk_overlap
        )
        splits = text_splitter.split_documents(loaded_docs)

        if vdb_type == "chroma":
            # TODO: From here, search for professional alternatives (cloud based vector databases ?)
            self.vector_db = Chroma.from_documents(
                documents=splits, embedding=self.embedding, persist_directory=persist_dir
            )
            self.vector_db.persist()
        else:
            raise NotImplementedError(f"Vector database type {vdb_type} not implemented")

    def load_vector_database(self, persist_dir, vdb_type="chroma"):
        if vdb_type == "chroma":
            self.vector_db = Chroma(persist_directory=persist_dir, embedding_function=self.embedding)
        else:
            raise NotImplementedError(f"Vector database type {vdb_type} not implemented")


    def search(self, query, k, search_type, filter_on=None):
        """
        :param query: The query to search for (input from the user in natural language)
        :param k: Number of relevant chunks to return across all document. If filter is set on a document, return k
        chunk in the doc.
        :param search_type: similarity or mmr
        :param filter_on: If set, filter the search on the document. The filter is a dictionary with one key. It can be
        either "source" or "page". (i.e. {"source":"docs/cs229_lectures/Lecture03.pdf"} or {"page": "1"})
        """
        assert search_type in ["similarity", "mmr"], f"search_type must in ['similarity', 'mmr'] got {search_type}"

        if search_type == "similarity":
            result = self.vector_db.similarity_search(query, k=k, filter=filter_on)
        else:  # mmr
            result = self.vector_db.max_marginal_relevance_search(query, k=k, filter=filter_on)

        return result


    def answer(self, query, search_result, chain_type="refine", rtp=None, verbose=False):
        """
        :param query: The query to search for (input from the user in natural language)
        :param search_result: Result of the search using "similarity" or "mmr" in self.search()
        :param chain_type: Either "stuff" or "refine"
        :param rtp: The RetrievalTemplate object
        :param verbose: verbose bro!!
        :return: The answer to the query
        """
        assert chain_type in ["stuff", "refine"], f"chain_type must in ['stuff', 'refine'] got {chain_type}"
        assert rtp is not None, "retrieval_template_obj must be provided"
        guess = ""

        if chain_type == "stuff":
            document_separator = "\n<<<<.>>>>\n"
            context = []
            for res in search_result:
                chunked_content = res.page_content
                context.append(chunked_content)

            context_str = document_separator.join(context)
            prompt_template = PromptTemplate(template=rtp.stuff_template, input_variables=["context", "question"])
            prompt = prompt_template.format(context=context_str, question=query)
            guess = self._run_inference(prompt)

            if verbose:
                print(f"Prompt\n {prompt}\n")
                print(f"Answer: {guess}\n")

        elif chain_type == "refine":
            # First guess
            first_context = search_result[0].page_content
            inputs = ["context", "question"]
            prompt_template = PromptTemplate(template=rtp.refine_template_start, input_variables=inputs)
            prompt = prompt_template.format(context=first_context, question=query)
            guess = self._run_inference(prompt)
            old_guess = guess

            if verbose:
                print(f"Prompt\n {prompt}\n")
                print(f"Guess 1: {guess}")

            # Refine the answer
            other_contexts = search_result[1:]

            if len(other_contexts) > 0:
                for n, next_context in enumerate(other_contexts):
                    next_context = next_context.page_content
                    inputs = ["question", "guess", "context"]
                    prompt_template = PromptTemplate(template=rtp.refine_template_next, input_variables=inputs)
                    prompt = prompt_template.format(context=next_context, question=query, guess=guess)
                    guess = self._run_inference(prompt)

                    guess_alpha_num = re.sub(r'\W+', '', guess)
                    if guess_alpha_num.strip() == "" or len(guess_alpha_num) <= 1:
                        guess = old_guess


                    if verbose:
                        print(f"Prompt\n {prompt}\n")
                        print(f"Guess {n + 2}: {guess}\n\n")

                if verbose: print(f"Final Answer: {guess}")

        return guess

    def _run_inference(self, prompt):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids
        model_output = self.llm.generate(input_ids=input_ids, generation_config=self.gen_config)
        response = self.tokenizer.decode(model_output[0], skip_special_tokens=True)
        return response


    @staticmethod
    def reset_persist_directory(persist_dir):
        if not isinstance(persist_dir, str):
            persist_dir = str(persist_dir)
        if os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)
















