import streamlit as st
import time
from rag_engine import ClimateRAG
from db_manager import DatabaseManager

st.set_page_config(page_title="Climate Agent", page_icon="🌍", layout="wide")


st.title("🌍 Climate News Agent")

# --- DATABASE MANAGEMENT SIDEBAR ---
with st.sidebar:
    st.header("Database Management")
    db_mgr = DatabaseManager()
    count = db_mgr.get_count()
    st.write(f"Current Records in Milvus: **{count}**")
    
    if st.button("Force Re-Sync Data"):
        with st.spinner("Syncing Parquet/NPY to Milvus..."):
            success, msg = db_mgr.ingest_default_data()
            if success:
                st.success(msg)
                st.rerun()
            else:
                st.error(msg)


# CHAT INTERFACE
# frontend.py
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