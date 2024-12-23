
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.global_setting import LLM_PATH


def get_llm_agent(llm_path=LLM_PATH):
    model = AutoModelForCausalLM.from_pretrained(llm_path, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(llm_path)
    tokenizer.pad_token_id = tokenizer.eos_token_id 
    return model, tokenizer
