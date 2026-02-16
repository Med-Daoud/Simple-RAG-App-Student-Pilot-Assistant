

from langchain_community.vectorstores import Chroma
from langchain_ollama import OllamaEmbeddings
from config import *

def get_retriever():
    embeddings = OllamaEmbeddings(model=EMBED_MODEL)

    vectorstore = Chroma(
        persist_directory=VECTOR_DB_PATH,
        embedding_function=embeddings
    )

    return vectorstore.as_retriever(search_kwargs={"k": TOP_K})
