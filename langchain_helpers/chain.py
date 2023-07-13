import os

import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.chains.router import MultiPromptChain
from langchain.chains.router.llm_router import LLMRouterChain, RouterOutputParser



if __name__ == "__main__":
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0)

    ###################################################################################################################

    prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
    chain = LLMChain(llm=llm, prompt=prompt)
    chain.run("Product A")  # the chain will format the prompt with the {product}, then run the llm.

    ###################################################################################################################

    # Other type of chain: Sequential chain (Single input, Single output)
    # > combine multiple chains where the output of one chain is the input of the next chain
    first_prompt = ChatPromptTemplate.from_template("What is the best name to describe a company that makes {product}?")
    first_chain = LLMChain(llm=llm, prompt=first_prompt)

    second_prompt = ChatPromptTemplate.from_template(
        "Write a 20 words description for the following company:{company_name}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt)
    # Combine
    overall_simple_chain = SimpleSequentialChain(chains=[first_chain, second_chain], verbose=True)
    overall_simple_chain.run("Product A")

    ###################################################################################################################

    # Other type of chain: Sequential chain (Multiple inputs, Multiple outputs)
    first_prompt = ChatPromptTemplate.from_template("Translate the following review to english:\n\n{Review}")
    first_chain = LLMChain(llm=llm, prompt=first_prompt, output_key="English_Review")

    second_prompt = ChatPromptTemplate.from_template(
        "Can you summarize the following review in 1 sentence:\n\n{English_Review}"
    )
    second_chain = LLMChain(llm=llm, prompt=second_prompt, output_key="summary")

    third_prompt = ChatPromptTemplate.from_template("What language is the following review:\n\n{Review}")
    third_chain = LLMChain(llm=llm, prompt=third_prompt, output_key="language")

    fourth_prompt = ChatPromptTemplate.from_template(
        "Write a follow up response to the following summary in the specified language:"
        "\n\nSummary: {summary}\n\nLanguage: {language}"
    )
    fourth_chain = LLMChain(llm=llm, prompt=fourth_prompt, output_key="followup_message")

    # Combine
    overall_chain = SequentialChain(
        chains=[first_chain, second_chain, third_chain, fourth_chain],
        input_variables=["Review"], output_variables=["English_Review", "summary", "followup_message"],
        verbose=True
    )
    overall_chain.run("Ce film etait magnifique, des acteurs qui savent jouer enfin à la hauteur de leur talent !")

    ###################################################################################################################

    # Router chain > a chain that will decide which sub-chain (your specialized chain) to use based on the input

    # setup multiple prompts - each prompt have its speciality (i.e. answering physics questions etc.)
    physics_template = """You are a very smart physics professor. \
    You are great at answering questions about physics in a concise\
    and easy to understand manner. \
    When you don't know the answer to a question you admit\
    that you don't know.

    Here is a question:
    {input}"""

    math_template = """You are a very good mathematician. \
    You are great at answering math questions. \
    You are so good because you are able to break down \
    hard problems into their component parts, 
    answer the component parts, and then put them together\
    to answer the broader question.

    Here is a question:
    {input}"""

    history_template = """You are a very good historian. \
    You have an excellent knowledge of and understanding of people,\
    events and contexts from a range of historical periods. \
    You have the ability to think, reflect, debate, discuss and \
    evaluate the past. You have a respect for historical evidence\
    and the ability to make use of it to support your explanations \
    and judgements.

    Here is a question:
    {input}"""

    computerscience_template = """ You are a successful computer scientist.\
    You have a passion for creativity, collaboration,\
    forward-thinking, confidence, strong problem-solving capabilities,\
    understanding of theories and algorithms, and excellent communication \
    skills. You are great at answering coding questions. \
    You are so good because you know how to solve a problem by \
    describing the solution in imperative steps \
    that a machine can easily interpret and you know how to \
    choose a solution that has a good balance between \
    time complexity and space complexity. 

    Here is a question:
    {input}"""

    # define useful information for the router chain in order to decide which sub-chain to use
    prompt_infos = [
        {
            "name": "physics",
            "description": "Good for answering questions about physics",
            "prompt_template": physics_template
        },
        {
            "name": "math",
            "description": "Good for answering math questions",
            "prompt_template": math_template
        },
        {
            "name": "History",
            "description": "Good for answering history questions",
            "prompt_template": history_template
        },
        {
            "name": "computer science",
            "description": "Good for answering computer science questions",
            "prompt_template": computerscience_template
        }
    ]

    # create a destination chain dictionary where the key is the name of the prompt and the value is the chain
    destination_chains = {}
    for p_info in prompt_infos:
        name = p_info["name"]
        prompt_template = p_info["prompt_template"]
        prompt = ChatPromptTemplate.from_template(prompt_template)
        chain = LLMChain(llm=llm, prompt=prompt)
        destination_chains[name] = chain

    # create a default chain that will be used if the router chain can't decide which sub-chain to use
    default_prompt = ChatPromptTemplate.from_template("{input}")
    default_chain = LLMChain(llm=llm, prompt=default_prompt)

    # define the template to use by the llm to route between the sub-chains
    MULTI_PROMPT_ROUTER_TEMPLATE = """Given a raw text input to a \
    language model select the model prompt best suited for the input. \
    You will be given the names of the available prompts and a \
    description of what the prompt is best suited for. \
    You may also revise the original input if you think that revising \
    it will ultimately lead to a better response from the language model.

    << FORMATTING >>
    Return a markdown code snippet with a JSON object formatted to look like:
    ```json
    {{{{
        "destination": string \ name of the prompt to use or "DEFAULT"
        "next_inputs": string \ a potentially modified version of the original input
    }}}}
    ```

    REMEMBER: "destination" MUST be one of the candidate prompt \
    names specified below OR it can be "DEFAULT" if the input is not\
    well suited for any of the candidate prompts.
    REMEMBER: "next_inputs" can just be the original input \
    if you don't think any modifications are needed.

    << CANDIDATE PROMPTS >>
    {destinations}

    << INPUT >>
    {{input}}

    << OUTPUT (remember to include the ```json)>>"""


    # create the router chain
    destinations = [f"{p['name']}: {p['description']}" for p in prompt_infos]
    destinations_str = "\n".join(destinations)

    router_template = MULTI_PROMPT_ROUTER_TEMPLATE.format(destinations=destinations_str)
    router_prompt = PromptTemplate(
        template=router_template, input_variables=["input"], output_parser=RouterOutputParser()
    )
    router_chain = LLMRouterChain.from_llm(llm, router_prompt)

    chain = MultiPromptChain(
        router_chain=router_chain, destination_chains=destination_chains, default_chain=default_chain, verbose=True
    )
    chain.run("what is 2 + 2")

    




