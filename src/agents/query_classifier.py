from src.prompts import CANDIDATE_LABELS
import difflib
from transformers import pipeline
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface.llms import HuggingFacePipeline
import torch


class QueryClassifier:
    def __init__(self, model, tokenizer, classification_prompt, labels = CANDIDATE_LABELS, device=None):
        # Handle device dynamically
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.labels = labels
        self.classification_prompt = classification_prompt

        # Define LLM chain using LangChain
        self.llm_completion_select_route_chain = (
            PromptTemplate.from_template(classification_prompt)
            | HuggingFacePipeline(
                pipeline=pipeline(
                    "text-generation",
                    model=model,
                    tokenizer=tokenizer,
                    torch_dtype="auto",
                    device=self.device,
                    max_new_tokens=2,  # Short output to match "labels"
                    num_return_sequences=1,
                    do_sample=False,
                )
            )
            | StrOutputParser()
        )

    def __call__(self, query):
        """
        Perform classification for a given query.
        """
        # Get raw LLM response
        raw_route_name = self.llm_completion_select_route_chain.invoke(query)

        # Extract the route name by splitting on "Answer:"
        route_name = raw_route_name.split('Answer:')[-1].strip()

        # Find the most similar label using difflib
        most_similar_label = difflib.get_close_matches(route_name, self.labels, n=1, cutoff=0.0)

        # Handle no match case
        if most_similar_label:
            return most_similar_label[0]
        else:
            return "No match found"