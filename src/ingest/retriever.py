from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder


class Retriever():
    def __init__(self, documents, top_k=5, embedding_name=None, reranker_name=None, embedding=None, reranker=None, device='cpu'):
        self.documents = documents
        self.embedding_name = embedding_name
        self.reranker_name = reranker_name
        self.top_k = top_k
        self.embedding = embedding
        self.reranker = reranker

    def load_embedding(self):
        if self.embedding is None:
            if self.embedding_name is not None:
                self.embedding = HuggingFaceEmbeddings(
                    model_name=self.embedding_name)
            else:
                self.embedding = HuggingFaceEmbeddings(
                    model_name="BAAI/bge-m3")

    def load_reranker(self):
        if self.reranker is None:
            if self.reranker_name is not None:
                self.reranker = HuggingFaceCrossEncoder(
                    model_name=self.reranker_name)
            else:
                self.reranker = HuggingFaceCrossEncoder(
                    model_name="BAAI/bge-reranker-base")

    def get_retriever(self):
        self.load_embedding()
        self.load_reranker()
        qdrant = Qdrant.from_documents(
            self.documents,
            self.embedding,
            location=":memory:",
            collection_name="reranker",
        )
        retriever = qdrant.as_retriever(search_kwargs={'k': 10})
        compressor = CrossEncoderReranker(
            model=self.reranker, top_n=self.top_k)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, base_retriever=retriever
        )
        return compression_retriever
