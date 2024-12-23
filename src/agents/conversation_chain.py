import torch
from transformers import pipeline
from langchain_core.runnables import RunnableParallel, RunnableLambda
from langchain_core.output_parsers.string import StrOutputParser
from operator import itemgetter
from langchain_huggingface.llms import HuggingFacePipeline
from src.prompts import CANDIDATE_LABELS
import re


class ConversationChain():
    def __init__(self,
                rag_prompt,
                general_prompt, 
                daily_prompt, 
                hypothetical_prompt,
                inferium_prompt, 
                compression_retriever, 
                query_classifier, 
                model, 
                tokenizer, 
                candidate_labels = CANDIDATE_LABELS, 
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")):
        self.device = device
        self.rag_prompt = rag_prompt
        self.general_prompt = general_prompt
        self.inferium_prompt = inferium_prompt
        self.compression_retriever = compression_retriever
        self.query_classifier = query_classifier
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.candidate_labels = candidate_labels

        # Initialize the LLM
        pipe = pipeline(
          "text-generation",
          model=model,
          tokenizer=tokenizer,
          torch_dtype="auto",
          device_map=self.device,
          # device=0,
          # max_new_tokens=512,
          early_stopping = True,
          do_sample=False,
          num_return_sequences=1,
          pad_token_id=tokenizer.eos_token_id,
          eos_token_id=tokenizer.eos_token_id,
          # length_penalty=1.2,
          num_beams = 2,
          return_full_text=False,
          )
        self.llm = HuggingFacePipeline(pipeline=pipe)


        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)

        def generate_hypothetic(question, temperature=0.7, max_new_tokens=256):
            prompt = hypothetical_prompt.format(question=question)
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
              messages, tokenize=False, add_generation_prompt=True)
            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
            )
            generated_ids = [
                output_ids[len(input_ids) :]
                for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            return self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        self.rag_chain = (
            RunnableParallel(
                question = lambda x: x["question"],
                chat_history= lambda x: x["chat_history"],
            )
            | RunnableParallel( # Modified line: Using RunnableLambda to wrap the logic
                  context = RunnableLambda(lambda x: generate_hypothetic(x["question"])) | compression_retriever | RunnableLambda(format_docs),
                  chat_history = itemgetter('chat_history'),
                  question = itemgetter('question')
            )
            | rag_prompt
            | self.llm
            | StrOutputParser()
        )
        self.inferium_chain = (
            RunnableParallel(
                question = lambda x: x["question"],
                chat_history= lambda x: x["chat_history"],
            )
            |RunnableParallel(
                  context = itemgetter ('question')| compression_retriever | format_docs,
                  chat_history = itemgetter ('chat_history') ,
                  question = itemgetter ('question'))
            | inferium_prompt
            | self.llm
            | StrOutputParser()
        )
        self.general_chain = (
            general_prompt
            | self.llm
            | StrOutputParser()
        )

        self.daily_chain = (
            daily_prompt
            | self.llm
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
            return "\n".join([f"Question: {chat_history[idx]['question']}\nAnswer: {chat_history[idx]['answer']}\n" for idx in range(len(chat_history) - 1, -1, -1)])


    def process_response(self, conversation):
        conversation = conversation.split('Answer:')[0]
        conversation = conversation.split('Question:')[0]
        conversation = conversation.split('Chat history:')[0]
        conversation = conversation.split('chat history:')[0]
        conversation = conversation.split('Context:')[0]
        conversation = conversation.split('context:')[0]

        from difflib import SequenceMatcher
        def calculate_similarity(a, b):
            return SequenceMatcher(None, a.lower(), b.lower()).ratio()

        def remove_duplicates(text, similarity_threshold=0.8):
            paragraphs = text.split('\n\n')
            
            unique_paragraphs = []
            
            for current_para in paragraphs:
                current_para = current_para.strip()
                is_duplicate = False
                
                for existing_para in unique_paragraphs:
                    similarity = calculate_similarity(current_para, existing_para)
                    
                    if similarity > similarity_threshold:
                        is_duplicate = True
                        break
                
                if not is_duplicate and current_para:
                    unique_paragraphs.append(current_para)
            
            return '\n\n'.join(unique_paragraphs)

        conversation = conversation.strip()
        conversation = remove_duplicates(conversation)
        return conversation
    
    def chat(self, message):
        query = message['question'].lower()
        chat_history = message['chat_history']

        if chat_history is None:
            chat_history = ""
        else:
            if isinstance(chat_history, str):
                chat_history = chat_history.lower()
            else:
                chat_history = [chat_history[idx] for idx in range(len(chat_history) - 1, -1, -1) if idx < 1]
                history = self.format_chat_history(chat_history).lower()
        if query == 'exit':
            print('Exiting')
            return 'Exiting'
        if query == '':
            return 'No query'
        label= self.query_classifier(query)
        candidate_labels = self.candidate_labels
        print(label)
        if label == candidate_labels[2]:
            output = self.daily_chain.invoke(query)
        elif label == candidate_labels[0]:
            output = self.rag_chain.invoke({"question": query, "chat_history": history})
        elif label == candidate_labels[-1]:
            output = self.inferium_chain.invoke({"question": query, "chat_history": history})
        else:
            output = self.general_chain.invoke({"question": query, "chat_history": history})
        response = self.process_response(output)

        return response