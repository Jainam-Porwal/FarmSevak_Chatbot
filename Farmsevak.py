import streamlit as st
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import os
from deep_translator import GoogleTranslator

load_dotenv()

# ---------------- Language Mapping ----------------
lang_map = {
    "English": "en",
    "Hindi": "hi",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Odia": "or"
}

# ---------------- LLM Setup ----------------
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

# Always generate in English, limit 150 words
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
Answer the farmer's question clearly in English.
Keep the answer short, maximum 150 words.
If you don't know, say so.

Question: {question}
Answer:
"""
)
chain = prompt | model | parser

# ---------------- Streamlit UI ----------------
st.set_page_config(page_title="🌾 FarmSevak Chatbot", page_icon="🌱")

st.image("Farm sevak.jpg", width=150)

# Default language
if "language" not in st.session_state:
    st.session_state.language = "English"

# Pre-translate static UI text when language changes
if "translations" not in st.session_state or st.session_state.last_lang != st.session_state.language:
    target = lang_map[st.session_state.language]
    def tr(text):
        if target == "en":
            return text
        try:
            return GoogleTranslator(source="en", target=target).translate(text)
        except Exception:
            return text

    st.session_state.translations = {
        "title": tr("🌾 FarmSevak"),
        "subtitle": tr("Your multilingual farming assistant 🌱"),
        "select_lang": tr("🌐 Select Language"),
        "ask": tr("Ask your farming question..."),
        "selected_lang": tr("✅ Selected Language:")
    }
    st.session_state.last_lang = st.session_state.language

# Show UI
st.title(st.session_state.translations["title"])
st.write(st.session_state.translations["subtitle"])
st.subheader(st.session_state.translations["select_lang"])

# Language buttons
col1, col2, col3, col4, col5 = st.columns(5)
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
with col5:
    if st.button("ଓଡ଼ିଆ"):
        st.session_state.language = "Odia"

st.write(f"{st.session_state.translations['selected_lang']} {st.session_state.language}")

# ---------------- Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a farmer assistant.")]

user_input = st.chat_input(st.session_state.translations["ask"])

if user_input:
    # Translate user input → English
    if st.session_state.language != "English":
        try:
            translated_input = GoogleTranslator(
                source=lang_map[st.session_state.language], target="en"
            ).translate(user_input)
        except Exception:
            translated_input = user_input
    else:
        translated_input = user_input

    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get LLM answer (English, ≤150 words)
    result = chain.invoke({"question": translated_input})

    # Translate answer → target language
    if st.session_state.language != "English":
        try:
            translated_result = GoogleTranslator(
                source="en", target=lang_map[st.session_state.language]
            ).translate(result)
        except Exception:
            translated_result = f"(⚠ Translation failed, showing English)\n\n{result}"
    else:
        translated_result = result

    st.session_state.chat_history.append(AIMessage(content=translated_result))

# ---------------- Display Chat ----------------
for msg in st.session_state.chat_history:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)
