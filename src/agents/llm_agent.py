
from transformers import pipeline
from langchain_huggingface.llms import HuggingFacePipeline
from src.global_setting import LLM_PATH


def get_llm_agent(device='cpu'):
    pipe = pipeline(
        "text-generation",
        model=LLM_PATH,
        tokenizer=LLM_PATH,
        torch_dtype="auto",
        device_map=device,
        max_new_tokens=512,
        do_sample=True,
        top_k=10,
        num_return_sequences=1,
    )
    llm = HuggingFacePipeline(pipeline=pipe)
    return llm
