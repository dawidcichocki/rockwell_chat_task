import os
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain, StuffDocumentsChain
from src.document_loader import PDFDocumentLoader
from src.vector_store import VectorStore
from src.llm_broker import LLMBroker
from langchain.prompts import PromptTemplate


class RAG:
    def __init__(self, file_paths: list[str]):
        self._file_paths = file_paths
        self._documents = []
        self._vector_store = None
        self._llm_broker = LLMBroker()
        self._conversation_chain = None
        self._system_prompt = os.getenv("SYSTEM_PROMPT")
        self._init_rag()

    def _init_rag(self):
        """
        Initializes the RAG model
        """
        document_loader = PDFDocumentLoader(self._file_paths)
        memory = ConversationBufferMemory(
            memory_key='chat_history', return_messages=True
        )
        self._documents = document_loader.load_documents()
        self._vector_store = VectorStore(self._documents)

        combine_docs_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                self._system_prompt +
                "\nUse the following documents to answer the question: "
                "{context} Question: {question}"
            )
        )

        question_gen_prompt = PromptTemplate(
            input_variables=["chat_history", "question"],
            template=(
                self._system_prompt +
                "\nBased on the following conversation, rephrase the query "
                "\"{question}\" to better extract information: {chat_history}"
            )
        )

        # Define LLMChain for combining documents
        combine_docs_llm_chain = LLMChain(
            llm=self._llm_broker.get_chat_model(),
            prompt=combine_docs_prompt
        )

        # Define StuffDocumentsChain with the LLMChain
        combine_docs_chain = StuffDocumentsChain(
            llm_chain=combine_docs_llm_chain,
            document_variable_name="context"  # Specify the document variable name
        )

        question_generator_chain = LLMChain(
            llm=self._llm_broker.get_chat_model(),
            prompt=question_gen_prompt
        )

        self._conversation_chain = ConversationalRetrievalChain(
            retriever=self._vector_store.get_retriever(),
            combine_docs_chain=combine_docs_chain,
            question_generator=question_generator_chain,
            memory=memory
        )

    def query(self, question: str):
        """
        Queries the RAG model with a question
        """
        response = self._conversation_chain({'question': question})
        return response