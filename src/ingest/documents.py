import pandas as pd
from langchain.docstore.document import Document
from PyPDF2 import PdfReader


def format_row(row):
    return (
        f"model name: {row['Name']}; "
        f"Link to the {row['Name']} model: {row['Link']};"
        f"{row['Name']} model task: {row['Task']}; "
        f"{row['Name']} specific task: {row['Specific task']}; "
        f"detail descriontion about {row['Name']} model: {row['Description']}; "
        f"this model based on: {str(row['Base Model'])};"
        f"detail evaluation of {row['Name']} model:"
        f"rating: {row['Stars']} star, "
        f"score: {row['Score']} ;"
        f"number of downloads: {row['Downloads']}; "
        f"advantanges of {row['Name']}: {row['Advantage']}; "
        f"disadvantages of {row['Name']}: {row['Disadvantage']};"
        f"Experiment result of {row['Name']} model based on metric {row['Metric']}: {row['Result']};"
        # f"In my website inferium, this model get input as {row['Input']} and return output as {row['Output']};"
    ).lower()
def get_documents_from_cvs(filepaths, format_row):
    documents = []
    chunk_size = 1024
    for filepath in filepaths:
        df = pd.read_csv(filepath)
        df['formatted_text'] = df.apply(format_row, axis=1)

        for row in df.itertuples():
            text = row.formatted_text
            model_name = row.Name
            chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
            
            for idx, chunk in enumerate(chunks):
                documents.append(Document(
                    page_content=chunk,
                    metadata={"chunk_index": idx, "topic":model_name, "source": filepath + "-" + model_name}
                ))
    return documents


def get_documents_from_pdf(filepaths):
    documents = []
    chunk_size = 512

    for filepath in filepaths:
        pdf_reader = PdfReader(filepath)
        text = ""

        for page in pdf_reader.pages:
            text += page.extract_text()

        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

        for idx, chunk in enumerate(chunks):
            documents.append(Document(
                page_content=chunk,
                metadata={"chunk_index": idx, "topic": "inferium-website", "source": filepath}
            ))

    return documents


def get_documents(filepaths):
    cvs_files = []
    pdf_files = []
    for filepath in filepaths:
        if filepath.endswith('.csv'):
            cvs_files.append(filepath)
        elif filepath.endswith('.pdf'):
            pdf_files.append(filepath)
    
    if len(cvs_files) > 0:
        documents_from_cvs = get_documents_from_cvs(cvs_files, format_row)
    if len(pdf_files) > 0:
        documents_from_pdf = get_documents_from_pdf(pdf_files)
    documents = documents_from_cvs + documents_from_pdf
    return documents