import streamlit as st
from dotenv import load_dotenv
from utils.ingestion import get_pdf_text, get_text_chunks
from utils.vectorstore import get_vectorstore
from utils.generation import get_conversation_chain
from config.settings import MEMORY_KEY

def handle_userinput(user_question):
    # Invoke conversation chain to get response and update memory
    response = st.session_state.conversation.invoke({'question': user_question})

    # Get updated chat history from conversation memory 
    st.session_state.conversation_messages = response[MEMORY_KEY]
    
    # Display full chat history
    for i, message in enumerate(st.session_state.conversation_messages):
        if i % 2 == 0:
            st.write("You: " + message.content)
        else:
            st.write("Bot: " + message.content)

def setup_session_state():
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "conversation_messages" not in st.session_state:
        st.session_state.conversation_messages = None

def process_pdfs(pdf_docs):
    raw_text = get_pdf_text(pdf_docs)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    st.session_state.conversation = get_conversation_chain(vectorstore)

def run_app():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon=":books:")
    setup_session_state()

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.header("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDF files", accept_multiple_files=True, type=["pdf"])
        if st.button("Process"):
            with st.spinner("Processing..."):
                process_pdfs(pdf_docs)

def main():
    load_dotenv()
    run_app()

if __name__ == "__main__":
    main()