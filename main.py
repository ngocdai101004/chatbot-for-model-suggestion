from langchain.prompts import PromptTemplate
from src.prompts import GENERAL_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE
from src.ingest.ingest import data_retriever
from src.agents.llm_agent import get_llm_agent
from src.agents.query_classifier import get_query_classifier, query_classification
from src.agents.conversation_chain import ChatbotChain
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_chain():
    retriever = data_retriever()

    general_prompt = PromptTemplate(
        input_variables=["chat_history", "question"],
        template=GENERAL_PROMPT_TEMPLATE
    )

    rag_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=RAG_PROMPT_TEMPLATE
    )

    query_classifier = get_query_classifier(device=device)

    llm = get_llm_agent(device=device)

    conversation_chain = ChatbotChain(llm=llm, rag_prompt=rag_prompt, general_prompt=general_prompt,
                                      compression_retriever=retriever, query_classifier=query_classifier)
    return conversation_chain


# Example
if __name__ == '__main__':
    conversation_chain = get_chain()
    while True:
        query = input ("Enter your query (enter 'exit' to exit): ")
        chat_history = [
            {
                'question': 'Hello',
                'answer': 'Hello, What can I help you'
            }
        ]
        message = {
            'question': query,
            'chat_history': chat_history
        }
        response = conversation_chain.chat(message)
        chat_history.append({
            'question': query,
            'answer': response
        })
        print("------------------Response-------------------\n", response)
        print("------------------History------------------- \n", len(chat_history))
        print("---------------------------------------------")
