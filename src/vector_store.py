from typing import List
import logging
from src.llm_broker import LLMBroker
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


class VectorStore:
    def __init__(self, documents: List[Document]):
        self._documents = documents
        self._model_broker = LLMBroker()
        self._vector_store = None
        self._init_vector_store()

    def _init_vector_store(self):
        """
        Stores the documents in the vector store using FAISS
        """
        embeddings = self._model_broker.get_embedding_model()
        self._vector_store = FAISS.from_documents(
            self._documents, embedding=embeddings
        )
        logging.info("Vector store initialized")

    def get_retriever(self):
        """
        Returns the retriever
        """
        return self._vector_store.as_retriever()

    def retrieve_with_scores(self, question: str, verbose: bool = True) -> List[Document]:
        """
        Retrieves the most relevant document for a given question
        """
        logging.info(f"Querying vector store with question: {question}")
        similar_documents = self._vector_store.similarity_search(question)

        if verbose:
            logging.info("Retrieved documents:")
            for doc, score in similar_documents:
                logging.info(
                    f"Document: {doc.metadata['id']}, Page: {doc.metadata['page']}, "
                    f"Score: {score}, Text: {doc.content[:100]}"
                )
        else:
            logging.info(f"Retrieved {len(similar_documents)} documents")

        return similar_documents