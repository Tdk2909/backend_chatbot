from langchain.prompts import MessagesPlaceholder
from langchain_core.prompts import ChatPromptTemplate


def ntt_prompt():
    prompt = ChatPromptTemplate.from_messages(
    [
        (
        "system",
        "answer based on real data, if you don't know the answer do not try to answer it"
        "when user asks somewhat vague question, suggest a few possible answers so thay can ask more "
        "do not answer questions outside the scope of education (except for personal information provided by the user)"
        "you are a chatbot from Nguyen Tat Thanh University, you only use Vietnamese"
        "you are a wise and helpful chat bot that can converse like a normal person with empathy and logic"
        "your mission is to answer user inquiries about Nguyen Tat Thanh University"
        ),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
    return prompt