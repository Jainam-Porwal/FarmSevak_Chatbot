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

# ---------------- Helper Functions ----------------
def translate_text(text, target_lang="en"):
    """Translate text to target language (for UI/output)."""
    try:
        if not text:
            return ""
        if target_lang == "en":
            return text
        return GoogleTranslator(source="en", target=target_lang).translate(text)
    except Exception:
        return text

def translate_to_english(text):
    """Translate farmer input into English (for LLM)."""
    try:
        if not text:
            return ""
        return GoogleTranslator(source="auto", target="en").translate(text)
    except Exception:
        return text

# ---------------- LLM Setup ----------------
llm = HuggingFaceEndpoint(repo_id="openai/gpt-oss-20b")
model = ChatHuggingFace(llm=llm)
parser = StrOutputParser()

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

# Language buttons
col1, col2, col3, col4, col5 = st.columns(5)
new_lang = st.session_state.language

with col1:
    if st.button("English"):
        new_lang = "English"
with col2:
    if st.button("हिंदी"):
        new_lang = "Hindi"
with col3:
    if st.button("ગુજરાતી"):
        new_lang = "Gujarati"
with col4:
    if st.button("मराठी"):
        new_lang = "Marathi"
with col5:
    if st.button("ଓଡ଼ିଆ"):
        new_lang = "Odia"

# Refresh translations if language changed
if new_lang != st.session_state.language or "translations" not in st.session_state:
    st.session_state.language = new_lang
    target = lang_map[st.session_state.language]

    def tr(text):
        return translate_text(text, target)

    # FarmSevak stays in English always ✅
    st.session_state.translations = {
        "title": "🌾 FarmSevak",
        "subtitle": tr("Your multilingual farming assistant 🌱"),
        "ask": tr("Ask your farming question..."),
        "selected_lang": tr("✅ Selected Language:")
    }

# Show UI
st.title(st.session_state.translations["title"])
st.write(st.session_state.translations["subtitle"])
st.write(f"{st.session_state.translations['selected_lang']} {st.session_state.language}")

# ---------------- Chat ----------------
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [SystemMessage(content="You are a farmer assistant.")]

user_input = st.chat_input(st.session_state.translations["ask"])

if user_input:
    # Translate user input → English
    translated_input = translate_to_english(user_input) if st.session_state.language != "English" else user_input

    # Store farmer’s original input
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Get LLM response (English, ≤150 words)
    result = chain.invoke({"question": translated_input})

    # Translate back to farmer’s language
    if st.session_state.language != "English":
        try:
            translated_result = translate_text(result, lang_map[st.session_state.language])
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
