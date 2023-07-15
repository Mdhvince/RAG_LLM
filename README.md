# LLM with Retrieval Augmented Generation

The code for this repo was motivated by the fact that the [Langchain](https://python.langchain.com/docs/get_started/introduction.html) 🦜️ have many working example
of how to augment a language model with various source of data using the OpenAI API.  
However, I am not sure the use of the OpenAI API is private enough when working with sensitive data (for companies).  

So I started developing a similar approach using Langchain yes but with open source models coming from [HuggingFace](https://huggingface.co/) 🤗  
This way I have all the flexibility to:
- Fine-tune the model (e.g with PEFT)
- Align the model with RLHF
- Use the model in a private way (no API)

For the moment, I have only implemented:  
- [x] Retrieval on my data using vector storage
- [x] Two chain types
  - Stuff
  - Refine
- [x] Prompt generation regarding the chain type
- [x] Memory (For follow-up conversation)
- [x] Prompt generation for memory
- [x] A Nice UI using [Streamlit](https://streamlit.io/) + a debug section to follow the LLM Thought process (Made from scratch)

I am still looking for more things to implement. The repo is not meant to be a full-fledge product, I am just having some fun.  

Note: Because of the limited hardware I have, I am using a small model (flan-t5-small) with no additional training. The goal is to have a working prototype of 
the things described above.
