import os
from langchain.embeddings import OpenAIEmbeddings


def get_embeddings():
    """
    A helper function to create an instance of OpenAIEmbeddings using the API key from environment variables.

    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings initialized with the API key.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    return OpenAIEmbeddings(openai_api_key=api_key)