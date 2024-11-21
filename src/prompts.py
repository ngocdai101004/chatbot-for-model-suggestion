CLASSIFY_PROMPT_TEMPLATE = """
Classify the following question into one of the categories: daily, models, general, knowledge, or inferium.
"daily" :"greeting, or general social daily real-life communication",
"models" :"inquiries for suggestions or expressing interest in artificial intelligence (ai), machine learning (ml), or deep learning (DL) models. It can contain a name or type of model.",
"general":"general chat, non-technical questions, or basic everyday inquiries",
"knowledge":"knowledge, theories, or concepts unrelated to models",
"inferium":"ask for inferium website information or has keywork inferium"
<question>
{question}
</question>
Answer with one word
Answer:
"""
HYPOTHETICAL_PROMPT_TEMPLATE = """
List a list of words to description the input sentence. It is useful for information retrival 
Return as format:
keyword1, keyword2,...,input sentence.
sentence: {question}
Answer:
"""

GENERAL_PROMPT_TEMPLATE = """
You are a assistant for a website. Only reply user message in short sentence based on history.
If you do not have information, please response you don't know.
Here is the chat history: {chat_history}
Question: {question}
Answer:
"""

RAG_PROMPT_TEMPLATE = """
You are a helpful assistant. Your task is to provide the best suggestions about models that users want to find based on their description or request.
Describe model with detail information.
Have to add model's url link to inferium web.
The following documents are relevant to the question: {context}
Here is the chat history: {chat_history}
Here is the user question: {question}
Answer:
"""

INFERIUM_PROMPT_TEMPLATE = """
You are a helpful assistant for inferium website.
Write a paragraph that answers the question with detail information.
Have to add url link to inferium web.
The following documents are relevant to the question: {context}
Here is the chat history: {chat_history}
Here is the user question: {question}
Answer:
"""


DAILY_PROMPT_TEMPLATE = """
You are a chatbot assistant for the inferium website. Respond concisely and only answer the user's question without providing any additional information or use cases.
Question: {question}
Answer:
"""

CANDIDATE_LABELS = ['models', 'general', 'daily', 'knowledge', 'inferium']