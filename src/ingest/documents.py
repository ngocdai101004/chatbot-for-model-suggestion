import requests
import pandas as pd
from langchain.docstore.document import Document
from src.ingest.retriever import Retriever


def format_row(row):
    return (
        f"model name: {row['Name'].lower()}; "
        f"model task: {row['Tasks'].lower()}; "
        f"detail descriontion about model: {row['Description']}; "
        f"detail evaluation of models:"
        f"rating: {row['Stars']} star, "
        f"score: {row['Score']} ;"
        f"number of downloads: {row['Downloads']}; "
        f"URL link to model in web: {row['Link_inferium']}; "
    ).lower()

def get_documents(filepaths, format_row=format_row):
    documents = []
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df['formatted_text'] = df.apply(format_row, axis=1)
        for text in df['formatted_text']:
            document = Document(page_content=text)
            documents.append(document)
    return documents
