#!/usr/bin/env python

from transformers import FillMaskPipeline, TFBertForMaskedLM, BertTokenizer

model = TFBertForMaskedLM.from_pretrained("bert-base-cased")
tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
pipeline = FillMaskPipeline(model, tokenizer, top_k=5)

def unmask(masked):
    """Print most likely substitutes for [MASK] tokens in a sentence"""
    print("Masked sentence:", masked)
    for unmasked in pipeline(masked):
        print(f"Score {unmasked['score']:.2f}:", unmasked["sequence"])

if __name__ == "__main__":
    unmask("I think the DL4NLP lecture is pretty [MASK].")
    # YOUR CODE HERE

    unmask("The nurse went for a vacation because [MASK] wanted to get some rest")
    unmask("The president went for a vacation because [MASK] wanted to get some rest")
