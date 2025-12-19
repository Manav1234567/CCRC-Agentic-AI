import streamlit as st
from rag_engine import ClimateRAG

st.set_page_config(page_title="Climate News Agent", page_icon="🌍")
st.title("🌍 Climate News RAG Agent")

# Initialize the RAG engine once and store it in the session
if "agent" not in st.session_state:
    with st.spinner("Initializing RAG Engine..."):
        st.session_state.agent = ClimateRAG()

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input
if prompt := st.chat_input("Ask about climate news..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                response = st.session_state.agent.ask(prompt)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error: {e}")