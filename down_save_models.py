from transformers import AutoModelForSequenceClassification, AutoTokenizer
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, AutoModelForSequenceClassification


def down_save_automodel(name, dir):
    model = AutoModel.from_pretrained(name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(name)
    path = os.path.join(dir, name.replace('/', '_'))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


def down_save_automodel_for_llm(name, dir):
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(name)
    path = os.path.join(dir, name.replace('/', '_'))
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)


def down_save_automodel_for_seq_reranker(name, dir):
    model = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype="auto")
    tokenizer = AutoTokenizer.from_pretrained(name)
    path = os.path.join(dir, name.replace('/', '_'))
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)


models_dir = 'src/models'
embedding_name = "BAAI/bge-m3"
reranker_name = "BAAI/bge-reranker-base"
llm_name = "Qwen/Qwen2.5-3B"
down_save_automodel(name=embedding_name, dir=models_dir)
down_save_automodel_for_seq_reranker(
    name=reranker_name, dir=models_dir)
down_save_automodel_for_llm(name=llm_name, dir=models_dir)
