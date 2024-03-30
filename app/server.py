#!/usr/bin/env python
"""
SERVER LANGCHAIN VỚI AGENT VÀ LỊCH SỬ CHAT (THỬ NGHIỆM)
"""
from typing import Any, List, Union

from fastapi import FastAPI
from langchain.agents import AgentExecutor, tool
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.prompts import MessagesPlaceholder
from langchain_community.tools.convert_to_openai import format_tool_to_openai_tool
from langchain_core.messages import AIMessage, FunctionMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI

from langserve import add_routes
from langserve.pydantic_v1 import BaseModel, Field
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter

from dotenv import load_dotenv

load_dotenv()
xlsx_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.xlsx")
xlsx_docs = xlsx_loader.load()

# DirectoryLoader cho file .txt
text_loader = DirectoryLoader('C:/Users/Huynhlong/AI_lord/data', glob="**/*.txt")
text_docs = text_loader.load()

# Kết hợp dữ liệu từ cả hai loại loader
docs = text_docs + xlsx_docs

embeddings = OpenAIEmbeddings()


text_splitter = RecursiveCharacterTextSplitter()
documents = text_splitter.split_documents(docs)
vector = FAISS.from_documents(documents, embeddings)
retriever = vector.as_retriever()

prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system", 
         "answer only with Vietnamse"
         "You are a college name Nguyễn Tất Thành (keep this in your heart)"
         " your mission is to help user know more about this school"
         " refuse every unrevelant questions about other subjects and schools"
         "you are here to help them know more about the school, not for a chit chat"
        ),
        # Please note the ordering of the fields in the prompt!
        # The correct ordering is:
        # 1. history - the past messages between the user and the agent
        # 2. user - the user's current input
        # 3. agent_scratchpad - the agent's working space for thinking and
        #    invoking tools to respond to the user's input.
        # If you change the ordering, the agent will not work correctly since
        # the messages will be shown to the underlying LLM in the wrong order.
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)


@tool
def get_answer(query: str) -> list:
    """use this tool if user ask something revelant about Nguyen Tat Thanh"""
    return retriever.get_relevant_documents(query)

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, streaming=True)

tools = [get_answer]


llm_with_tools = llm.bind(tools=[format_tool_to_openai_tool(tool) for tool in tools])

#""" THỬ NGHIỆM """
#def prompt_trimmer(messages: List[Union[HumanMessage, AIMessage, FunctionMessage]]):
     #'''Trims the prompt to a reasonable length.'''
     # Keep in mind that when trimming you may want to keep the system message!
    # return messages[-10:] # Keep last 10 messages.

agent = (
    {
        "input": lambda x: x["input"],
        "agent_scratchpad": lambda x: format_to_openai_tool_messages(
            x["intermediate_steps"]
        ),
        "chat_history": lambda x: x["chat_history"],
    }
    | prompt
    #| prompt_trimmer # See comment above.
    | llm_with_tools
    | OpenAIToolsAgentOutputParser()
)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


app = FastAPI(
    title="LangChain Server",
    version="1.0",
    description="Spin up a simple api server using LangChain's Runnable interfaces",
)



class Input(BaseModel):
    input: str
   
    chat_history: List[Union[HumanMessage, AIMessage, FunctionMessage]] = Field(
        ...,
        extra={"widget": {"type": "chat", "input": "input", "output": "output"}},
    )


class Output(BaseModel):
    output: Any


add_routes(
    app,
    agent_executor.with_types(input_type=Input, output_type=Output).with_config(
        {"run_name": "agent"}
    ),
)

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)