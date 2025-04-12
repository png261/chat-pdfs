from dotenv import load_dotenv
from Agent import Agent
from htmlTemplates import css, bot_template, user_template
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()


def get_pdf_text(pdf_docs) -> str:
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()

    return text


def get_text_chunks(raw_text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(raw_text)
    return chunks


def get_vectorsore(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001")
    vectorstores = Chroma.from_texts(
        texts=text_chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"  # Optional: save to disk
    )
    return vectorstores


def handle_userinput(user_question):
    # Initialize session state variables if not already set
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    try:
        st.session_state.chat_history.append(user_question)

        response = st.session_state.agent.query(user_question)
        st.session_state.chat_history.append(response)

        # Display the conversation history
        for index, message in enumerate(st.session_state.chat_history):
            if index % 2 == 0:
                st.write(
                    user_template.replace("{{MSG}}", message),
                    unsafe_allow_html=True,
                )
            else:
                st.write(
                    bot_template.replace("{{MSG}}", message),
                    unsafe_allow_html=True,
                )

    except Exception as e:
        st.error(f"An error occurred: {str(e)}")


def main():
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "agent" not in st.session_state:
        st.session_state.agent = Agent()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on process",
            accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                vectorstore = get_vectorsore(text_chunks)
                st.session_state.agent.set_retriever(
                    vectorstore.as_retriever())


if __name__ == "__main__":
    main()
