GENERAL_PROMPT_TEMPLATE = """
You are a helpful assistant of inferium web. Only reply user message in short and friendly sentence based on history.
Here is the chat history: {chat_history}
Question: {question}
Answer: 
"""
RAG_PROMPT_TEMPLATE = """
You are a helpful assistant of inferium web . Your task is to provide the best suggestions about models that users want to find based on their description or request.
Describe model with detail information.
Have to add model's url link to inferium web.
The following documents are relevant to the question: {context}
Here is the chat history: {chat_history}
Here is the user question: {question}
Answer:
"""

CANDIDATE_LABELS = [
    "requesting suggestions or expressing interest in artificial intelligence (ai), machine learning (ml), or deep learning (DL) models, techniques, or implementations, especially in the context of our forum.",
    "general chat, such as greetings, social or non-technical questions, or basic everyday inquiries",
    "general inquiries about knowledge, theories, or concepts unrelated to models",
]