#!/usr/bin/env python

from matplotlib.pyplot import axis
from transformers import BertTokenizer, TFBertModel

import tensorflow as tf
from keras.utils import losses_utils
from keras.utils import tf_utils
#from tensorflow.keras.losses import cosine_similarity

def embed(text):
    model = TFBertModel.from_pretrained('bert-base-cased')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')

    # add BERT special tokens and tokenize
    marked_text = f"[CLS] {text} [SEP]"
    tokenized_text = tokenizer.tokenize(marked_text)

    # map token strings to vocabulary indeces
    input_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
    token_type_ids = [1] * len(tokenized_text)

    # convert inputs to tensors
    input_id_tensors = tf.constant([input_ids])
    token_type_tensors = tf.constant([token_type_ids])

    # get BERT representations from last hidden layer
    outputs = model(input_id_tensors, token_type_tensors)
    return tokenized_text, outputs.last_hidden_state

def compare_homonyms(tokens, embeddings):
    bank_indexes = [idx for idx, subword in enumerate(tokens) if subword == "bank"]
    # YOUR CODE HERE
    loss = tf.keras.losses.CosineSimilarity(
        axis =-1,
        reduction=losses_utils.ReductionV2.AUTO,
        name='cosine_similarity'

    )
    instance1 = loss(embeddings[0][6], embeddings[0][10]).numpy()
    instance2 = loss(embeddings[0][6], embeddings[0][21]).numpy()
    instance3 = loss(embeddings[0][10], embeddings[0][21]).numpy()

    print(instance1, instance2, instance3)
    #print(bank_indexes)


if __name__ == "__main__":
    # sentence with multiple meanings of the word bank
    text = "After stealing money from the bank vault, the bank robber was seen fishing on the Mississippi river bank."
    tokenized, embeddings = embed(text)

    print("Tokens:", tokenized)
    #print("embeddings:", embeddings[0][3], embeddings[0][6])
    compare_homonyms(tokenized, embeddings)
