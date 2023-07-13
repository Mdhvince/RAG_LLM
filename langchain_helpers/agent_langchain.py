import os
from datetime import date

import openai
from langchain.agents import load_tools, initialize_agent, AgentType, tool
from langchain.chat_models import ChatOpenAI


# with the @tool decorator, we can turn any function into a tool that can be used by the agent
# the docstring is important because the agent will use it to know when and how it should call the tool
@tool
def custom_tool_that_gives_the_time(text: str) -> str:
    """Returns today's date, use this for any questions related to today's date. The input should always be an empty \
    string, and this function will always return today's date - any date mathematics should occur outside this function.
    """
    return str(date.today())


if __name__ == "__main__":
    # Define the LLM
    openai.api_key = os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(temperature=0)

    # Define the tools
    list_tools = ["llm-math", "wikipedia"]
    tools = load_tools(list_tools, llm=llm) + [custom_tool_that_gives_the_time]

    # Initialize the agent
    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
        handle_parsing_errors=True,
        verbose=True,
    )
    print(agent("What is 25% of 300 ?"))

    print(agent("What's today's date ?"))




