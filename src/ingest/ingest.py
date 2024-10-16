import requests
import pandas as pd
from langchain.docstore.document import Document
from src.ingest.retriever import Retriever
from src.ingest.documents import get_documents
from src.global_setting import FILE_PATHS, EMBEDDING_PATH, RERANKER_PATH


def data_retriever():
    documents = get_documents(FILE_PATHS)
    retriever = Retriever(
        documents, embedding_name=EMBEDDING_PATH, reranker_name=RERANKER_PATH)
    return retriever.get_retriever()
