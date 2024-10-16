from transformers import pipeline
from src.global_setting import BART_PATH
from src.prompts import CANDIDATE_LABELS


def get_query_classifier(device='cpu'):
    classifier = pipeline("zero-shot-classification",
                          model=BART_PATH,
                          tokenizer=BART_PATH,
                          device_map=device,)
    return classifier


def query_classification(query, classifier, candidate_labels=CANDIDATE_LABELS):
    clss = classifier(query, candidate_labels)
    score = clss['scores'][0]
    return score
