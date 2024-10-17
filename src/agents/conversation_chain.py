from transformers import pipeline
from langchain.chains import LLMChain
from langchain_core.runnables import RunnableParallel
from langchain_core.output_parsers.string import StrOutputParser
from operator import itemgetter
from src.agents.query_classifier import query_classification
from src.prompts import CANDIDATE_LABELS


class ChatbotChain():
    def __init__(self, llm, rag_prompt, general_prompt, compression_retriever, query_classifier):
        self.llm = llm
        self.rag_prompt = rag_prompt
        self.general_prompt = general_prompt
        self.compression_retriever = compression_retriever

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        self.rag_chain = (
            RunnableParallel(
                question=lambda x: x["question"],
                chat_history=lambda x: x["chat_history"],
            )
            | RunnableParallel(
                context=itemgetter(
                    'question') | compression_retriever | format_docs,
                chat_history=itemgetter('chat_history'),
                question=itemgetter('question'))
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        self.general_chain = (
            general_prompt
            | llm
            | StrOutputParser()
        )

        self.query_classifier = query_classifier

    def set_rag_prompt(self, rag_prompt):
        self.rag_prompt = rag_prompt

    def set_general_prompt(self, general_prompt):
        self.general_prompt = general_prompt

    def set_compression_retriever(self, compression_retriever):
        self.compression_retriever = compression_retriever

    def set_llm(self, llm):
        self.llm = llm

    def set_query_classifier(self, query_classifier):
        self.query_classifier = query_classifier

    def format_chat_history(self, chat_history):

        if chat_history is None or len(chat_history) == 0:
            return ""
        else:
            return "\n".join([f"User: {record['question']}\nAssistant: {record['answer']}" for record in chat_history])

    def chat(self, message):
        query = message['question']
        chat_history = message['chat_history']
        chat_history = chat_history[len(chat_history)-2:len(chat_history)] if len(chat_history) > 2 else chat_history
        history = self.format_chat_history(chat_history)
        if query == 'exit':
            print('Exiting')
            return 'Exiting'
        if query == '':
            return 'No query'
        label = query_classification(
            query=query, classifier=self.query_classifier)
        if label== CANDIDATE_LABELS[0]:
            output = self.rag_chain.invoke(
                {"question": query, "chat_history": history})
            response = output.split('Answer:')[-1]
        else:
            output = self.general_chain.invoke(
                {"question": query, "chat_history": history})
            response = output.split('Answer:')[-1]
        return response
