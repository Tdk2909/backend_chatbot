from typing import Any

from data_verify import load_documents
from dotenv import load_dotenv
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import \
    format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import \
    OpenAIToolsAgentOutputParser
from langchain_community.vectorstores import FAISS
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.utils.function_calling import convert_to_openai_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langserve.pydantic_v1 import BaseModel, Field
from prompt_templates import ntt_prompt
from session_verify import create_session_factory

load_dotenv()

embeddings = OpenAIEmbeddings()


docs = load_documents()

prompt = ntt_prompt()

text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()


@tool
def ntt_tool(query: str) -> list:
    """use this tool if user ask something revelant about Nguyen Tat Thanh"""
    return retriever.invoke(query) # type: ignore


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.5, streaming=True)

tools = [ntt_tool]


llm_with_tools = llm.bind(tools=[convert_to_openai_tool(tool) for tool in tools])


agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(x["intermediate_steps"]),
        "chat_history": lambda x: x["chat_history"],
       
    }
    | prompt
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

class Input(BaseModel):
    input: str = Field(
    
        ...,
        desscription="The human input to the chat system.",
        extra={"widget": {"type": "chat", "input": "input", "output":"output"}},
    )



class Output(BaseModel):
    output: Any

chain_with_history_and_agent = RunnableWithMessageHistory(
    agent_executor,
    create_session_factory("chat_histories"),
    input_messages_key="input",
    history_messages_key="chat_history",
).with_types(input_type=Input)
