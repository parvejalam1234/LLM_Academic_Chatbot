
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

import streamlit as st
import os
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

msgs = StreamlitChatMessageHistory(key="langchain_messages")
if len(msgs.messages) == 0:
    msgs.add_ai_message("How can I help you?")

view_messages = st.expander("View the message contents in session state")


# Access environment variables
google_api_key = os.getenv("GOOGLE_API_KEY")
langchain_api_key = os.getenv("LANGCHAIN_API_KEY")

if not google_api_key or not langchain_api_key:
    st.error("API keys are missing. Please check your .env file.")

os.environ["GOOGLE_API_KEY"]=google_api_key
os.environ["LANGCHAIN_TRACING_V2"]="true"
os.environ["LANGCHAIN_API_KEY"]=langchain_api_key

# prop template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system","You're an assistant knowledgeable about academic. Only answer academic-related questions.As per user need you have provide answer based on previous chat."),
        MessagesPlaceholder(variable_name="history"),
        ("user", "Question:{question}")
    ]
)


llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro") 
output_parser = StrOutputParser()
chain = prompt | llm | output_parser

chain_with_history = RunnableWithMessageHistory(
    chain,
    lambda session_id: msgs,
    input_messages_key="question",
    history_messages_key="history",
)

for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if prompt := st.chat_input():
    st.chat_message("user").write(prompt)
    config = {"configurable": {"session_id": "any"}}
    response = chain_with_history.invoke({"question": prompt}, config)
    st.chat_message("ai").write(response)

with view_messages:
    view_messages.json(st.session_state.langchain_messages)
