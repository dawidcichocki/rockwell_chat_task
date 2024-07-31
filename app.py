import streamlit as st
from dotenv import load_dotenv
from glob import glob
from src.rag import RAG
from htmlTemplates import css, bot_template, user_template


def init_rag_instance(file_paths):
    return RAG(file_paths)


def save_pdfs(pdf_docs):
    for pdf_doc in pdf_docs:
        with open(f'rag_files/{pdf_doc.name}', "wb") as f:
            f.write(pdf_doc.getbuffer())


def get_file_paths():
    return glob("rag_files/*.pdf")


def handle_userinput(user_question):
    response = st.session_state.conversation.query(user_question)
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv('.env.example')
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "user_question" not in st.session_state:
        st.session_state.user_question = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    st.session_state['user_question'] = user_question

    if st.session_state['user_question']:
        handle_userinput(st.session_state['user_question'])
        st.session_state['user_question'] = ""

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # reset the conversation
                st.session_state.chat_history = None
                st.session_state.conversation = None
                if pdf_docs:
                    # save the pdfs
                    save_pdfs(pdf_docs)

                file_paths = get_file_paths()
                st.session_state.conversation = init_rag_instance(file_paths)
                st.success("Documents in rag_files folder processed successfully!")

                if pdf_docs:
                    # cleanup
                    pdf_docs.clear()


if __name__ == '__main__':
    main()