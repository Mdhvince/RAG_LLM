import time
from copy import copy, deepcopy
from pathlib import Path

import streamlit as st
import pandas as pd
from langchain import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings

try:
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, GenerationConfig, AutoModelForCausalLM
except ImportError:
    st.warning("Please reload the page.", icon=":warning:")

import utils
from search.cogninova_memory import CogninovaMemory
from search.cogninova_search import CogninovaSearch
from search.cogninova_template import CogninovaTemplate


embedding = HuggingFaceEmbeddings()
debug_filepath = "debug.txt"


@st.cache_resource
def load_model(model_name):
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto")
    cm = CogninovaMemory()
    return llm, cm


@st.cache
def load_vectors(cs, persist_dir, vdb_type):
    cs._load_vector_database(persist_dir, vdb_type=vdb_type)

def read_debug():
    text = ""
    if Path(debug_filepath).exists():
        with open(debug_filepath, "r") as f:
            text = f.read().replace("\n", "<br>")
    return text


def reload():
    return st.experimental_rerun


def highlight_memory(s):
    return ['background-color: #72CA9B; color:black' if i >= len(s)-2 else '' for i in range(len(s))]


if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Cogninova", page_icon=":taco:", initial_sidebar_state="expanded")
    assistant_avatar = "medias/avatar.png"
    user_avatar = "medias/avatar_user.png"
    cfg, cfg_vs, cfg_search, cfg_model = utils.get_config()

    # Setting up the sidebar
    with st.sidebar:
        p_bar = st.progress(0)

        with st.expander("#### :wrench: Configuration"):

            st.markdown("<center><strong>Search configuration</strong></center>", unsafe_allow_html=True)
            search_type = st.selectbox("Search type", ("Similarity", "MMR"), index=0)
            search_type = search_type.lower()
            k_search = st.slider("K-search", min_value=1, max_value=15, value=2, step=1)

            st.markdown("<center><strong>Model configuration</strong></center>", unsafe_allow_html=True)
            chain_type = st.selectbox("Chain type", ("Stuff", "Refine"), index=1)
            chain_type = chain_type.lower()

            disable = False
            top_k = st.slider("Top k", min_value=1, max_value=50, value=5, step=1, disabled=disable)
            top_p = st.slider("Top p", min_value=0.1, max_value=1.0, value=0.95, step=0.05, disabled=disable)
            mnt = st.slider("Max new tokens", min_value=10, max_value=1000, value=200, step=10, disabled=disable)
            temp = st.slider("Temperature", min_value=0.1, max_value=2.0, value=0.1, step=0.01, disabled=disable)
            nbs = st.slider("Beam width", min_value=1, max_value=3, value=2, step=1, disabled=disable)

        with st.expander("#### :ladybug: Debug"):
            debug_placeholder = st.empty()
            debug_placeholder.write(read_debug(), unsafe_allow_html=True)

        with st.expander("#### :brain: Working Memory"):
            memory_placeholder = st.empty()
            memory_placeholder.write("No memory yet.", unsafe_allow_html=True)

        clear_history = st.button(":wastebasket: Clear history", use_container_width=True)

        # Rewriting into the config after value change from the UI
        cfg_search["search_type"] = f"{search_type}"
        cfg_search["chain_type"] = f"{chain_type}"
        cfg_search["k_return"] = f"{k_search}"
        cfg_model["top_k"] = f"{top_k}"
        cfg_model["top_p"] = f"{top_p}"
        cfg_model["max_new_tokens"] = f"{mnt}"
        cfg_model["temperature"] = f"{temp}"
        cfg_model["num_beams"] = f"{nbs}"

    # Setting up the brain
    ct = CogninovaTemplate()
    llm, cm = load_model(cfg_model.get("name"))
    cs = CogninovaSearch(cfg_model, cfg_search, cfg_vs, llm, embedding)

    debug_placeholder.write(read_debug(), unsafe_allow_html=True)

    # Setting up the chat
    query = st.chat_input()

    if "messages" not in st.session_state:
        st.session_state["messages"] = []


    if clear_history and "messages" in st.session_state:
        st.session_state.messages = []
        cm.clear()
        with open(debug_filepath, "w") as f: f.write("")
        debug_placeholder.write(read_debug(), unsafe_allow_html=True)


    for msg in st.session_state.messages:
        st.chat_message(msg["role"], avatar=msg["avatar"]).write(msg["content"], unsafe_allow_html=True)

    # Let's chat
    if query:
        # Query
        with st.chat_message("user", avatar=user_avatar):
            st.session_state.messages.append({"role": "user", "content": query, "avatar": user_avatar})
            st.write(query, unsafe_allow_html=True)

        cm.update(query)
        memory_for_ui = list(cm.memory)

        if cm.is_full():
            chat_history: str = cm.get_chat_history()
            p_template = PromptTemplate(template=ct.standalone_question_template, input_variables=["chat_history"])
            prompt = p_template.format(chat_history=chat_history)
            query = cs.run_inference(prompt)
            memory_for_ui.append("Standalone question: " + query)
            cm.optimize(query)

        # Search
        search_result = cs.search(query, filter_on=None)

        # Answer
        with st.chat_message("assistant", avatar=assistant_avatar):
            message_placeholder = st.empty()
            full_response = ""

            with st.spinner(":thinking_face:"):
                natural_answer = cs.answer(query, search_result, template_obj=ct)
            cm.update(natural_answer)

            st.session_state.messages.append(
                {"role": "assistant", "content": natural_answer, "avatar": assistant_avatar})


            array_response = natural_answer.split()
            for n, chunk in enumerate(array_response):
                full_response += chunk + " "
                time.sleep(0.05)
                message_placeholder.markdown(full_response + "▌")
                p_bar.progress((n + 1) / len(array_response))

            message_placeholder.markdown(full_response)
            memory_for_ui.append("Assistant: " + natural_answer)

        debug_placeholder.write(read_debug(), unsafe_allow_html=True)


    with st.sidebar:
        try:
            df = pd.DataFrame(memory_for_ui, columns=['Memory'])
        except NameError:
            df = pd.DataFrame([], columns=['Memory'])
        with st.expander("#### :brain: Working Memory"):
            df = df.style.apply(highlight_memory)
            memory_placeholder.dataframe(df, use_container_width=True, hide_index=True)



