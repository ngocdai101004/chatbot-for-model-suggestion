from langchain.prompts import PromptTemplate
from src.prompts import GENERAL_PROMPT_TEMPLATE, RAG_PROMPT_TEMPLATE, INFERIUM_PROMPT_TEMPLATE, DAILY_PROMPT_TEMPLATE, CANDIDATE_LABELS, HYPOTHETICAL_PROMPT_TEMPLATE, CLASSIFY_PROMPT_TEMPLATE
from src.global_setting import LLM_PATH
from src.ingest.ingest import data_retriever
from src.agents.llm_agent import get_llm_agent
from src.agents.query_classifier import QueryClassifier
from src.agents.conversation_chain import ConversationChain
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

    inferium_prompt = PromptTemplate(
        input_variables=["chat_history", "question", "context"],
        template=INFERIUM_PROMPT_TEMPLATE
    )

    daily_prompt = PromptTemplate(
        input_variables=["question"],
        template=DAILY_PROMPT_TEMPLATE
    )

    model, tokenizer = get_llm_agent(llm_path=LLM_PATH)
    query_classifier = QueryClassifier(model=model,
                                       tokenizer=tokenizer,
                                       classification_prompt=CLASSIFY_PROMPT_TEMPLATE,
                                       labels=CANDIDATE_LABELS,
)



    conversation_chain =  ConversationChain( rag_prompt, 
                                            general_prompt, 
                                            daily_prompt, 
                                            HYPOTHETICAL_PROMPT_TEMPLATE, 
                                            inferium_prompt, 
                                            retriever, 
                                            query_classifier,
                                            model, tokenizer, 
                                            CANDIDATE_LABELS)
    return conversation_chain


# Example
if __name__ == '__main__':
    conversation_chain = get_chain()
    while True:
        query = input ("Enter your query (enter 'exit' to exit): ")
        if (query == 'exit'):
            break 
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
