import streamlit as st
import time
from rag_engine import ClimateRAG
from db_manager import DatabaseManager

st.set_page_config(page_title="Climate Agent", page_icon="🌍", layout="wide")


# --- MAIN APP: Auto-Hydrate Logic ---
st.title("🌍 Climate News Agent")


# CHAT INTERFACE
# Only load the RAG engine if we actually have data
if "agent" not in st.session_state:
    with st.spinner("Loading AI Models..."):
        st.session_state.agent = ClimateRAG()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about climate news..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching & Thinking..."):
            try:
                response = st.session_state.agent.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")