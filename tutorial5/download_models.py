#!/usr/bin/env python

from transformers import TFBertModel

# optional: download all models we might need before the tutorial
for model in ['bert-base-cased', 'bert-base-uncased', 'bert-base-multilingual-cased']:
    TFBertModel.from_pretrained(model)
