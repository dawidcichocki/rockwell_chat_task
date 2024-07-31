import os
import logging
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


class LLMBroker:
    def __init__(self):
        self._openai_chat_model = ChatOpenAI(
            model=os.getenv('OPENAI_CHAT_MODEL', 'gpt-3.5-turbo'),
            temperature=os.getenv('CHAT_LLM_TEMPERATURE', 0.5),
            max_tokens=os.getenv('CHAT_LLM_MAX_TOKENS', 1000),
            api_key=os.getenv('OPENAI_API_KEY')
        )
        self._openai_embeddings = OpenAIEmbeddings(
            model=os.getenv('OPENAI_EMBEDDINGS_MODEL', 'text-embedding-3-small'),
            dimensions=os.getenv('OPENAI_EMBEDDINGS_DIMENSIONS', 512),
            api_key=os.getenv('OPENAI_API_KEY')
        )
        logging.info("LLMs initialized")

    def get_chat_model(self):
        return self._openai_chat_model

    def get_embedding_model(self):
        return self._openai_embeddings