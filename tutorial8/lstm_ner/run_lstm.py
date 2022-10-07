#!/usr/bin/env python
import nltk
from util.preprocessing import addCharInformation, createMatrices, addCasingInformation
from neuralnets.BiLSTM import BiLSTM
from nltk.tokenize import word_tokenize


modelPath = "/home/wilfried/Downloads/Tati/Bielefeld/dl4nlp/tutorial/tutorial8/lstm_ner/EN_NER.h5"
inputPath = "input.txt"

# Read input
with open(inputPath, 'r') as f:
    raw_text = f.read()

# Load the model
lstmModel = BiLSTM.loadModel(modelPath)

# Prepare the input
sentences = word_tokenize(raw_text) # YOUR CODE HERE

# Do some library specific stuff
sentences = [{"tokens": sent} for sent in sentences]
addCharInformation(sentences)
addCasingInformation(sentences)

# Tag the input
tags = lstmModel.tagSentences(createMatrices(sentences, lstmModel.mappings, True))

# Output to stdout
for sentenceIdx, sentence in enumerate(sentences):
    tokens = sentence['tokens']

    for tokenIdx, token in enumerate(tokens):
        tokenTags = []
        for modelName in sorted(tags.keys()):
            tokenTags.append(tags[modelName][sentenceIdx][tokenIdx])

        print("%s\t%s" % (token, "\t".join(tokenTags)))
    print("")
