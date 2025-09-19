import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os

load_dotenv()

# Load model
llm = HuggingFaceEndpoint(repo_id="deepseek-ai/DeepSeek-V3.1")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Prompt
prompt = PromptTemplate(
    input_variables=["question", "language"],
    template="""
Answer the farmer's question clearly.
Always reply in {language}. If you don't know, say so.

Question: {question}
Answer (in {language}):
"""
)
chain = prompt | model | parser

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🌾 FarmSevak Chatbot", page_icon="🌱")

# Display logo
st.image("Farm sevak.jpg", width=150)
st.title("🌾 FarmSevak")
st.write("Your multilingual farming assistant 🌱")

# Language selection buttons
st.subheader("🌐 Select Language")
col1, col2, col3, col4 = st.columns(4)
if "language" not in st.session_state:
    st.session_state.language = "English"

with col1:
    if st.button("English"):
        st.session_state.language = "English"
with col2:
    if st.button("हिंदी"):
        st.session_state.language = "Hindi"
with col3:
    if st.button("ગુજરાતી"):
        st.session_state.language = "Gujarati"
with col4:
    if st.button("मराठी"):
        st.session_state.language = "Marathi"

st.write(f"✅ Selected Language: **{st.session_state.language}**")

# Session state for chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a farmer assistant.")]

# Chat input
user_input = st.chat_input("Ask your farming question...")

if user_input:
    # Append user input
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get answer
    result = chain.invoke({"question": user_input, "language": st.session_state.language})
    st.session_state.chat_history.append(AIMessage(content=result))

# Display chat
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
