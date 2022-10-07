#!/usr/bin/env python

from itertools import chain

import tensorflow as tf
from transformers import BertConfig, BertTokenizer, TFBertModel


def embed(sent, model, tokenizer, align_layer=8):
    """
    Embed sent and return a mapping from subwords to actual words. By default
    extract embeddings from 8th Bert layer which has been shown to work the
    best for alignment tasks (see https://aclanthology.org/2021.eacl-main.181).
    """

    # split the sentence on whitespace
    words = tokenizer.basic_tokenizer.tokenize(sent)
    # tokenize into subwords (which potentially splits words further)
    tokens = [tokenizer.tokenize(word) for word in words]
    ids = [tokenizer.convert_tokens_to_ids(x) for x in tokens]

    model_inputs = tokenizer.prepare_for_model(list(chain(*ids)), return_tensors='tf', truncation=True)
    embeddings = model(**{k: tf.expand_dims(v, 0) for k, v in model_inputs.items()})["hidden_states"][align_layer]

    # create a mapping from subwords to actual words
    sub2word_map = list()
    for word_index, word_list in enumerate(tokens):
        sub2word_map.extend([word_index] * len(word_list))

    # remove embeddings for [CLS] and [SEP] tokens as we don't need them
    return embeddings[0, 1:-1], words, sub2word_map

def align(src, tgt, model, tokenizer, threshold=1e-3):
    """
    Align words in src to words in tgt. This approach is based on
    https://aclanthology.org/2021.eacl-main.181.
    """
    embeddings_src, words_src, sub2word_map_src = embed(src, model, tokenizer)
    embeddings_tgt, words_tgt, sub2word_map_tgt = embed(tgt, model, tokenizer)

    # YOUR CODE HERE (follow the instructions)

    # Compute similarity of each src subword with all tgt subwords using the
    # dot-product on their word embeddings. Create a similarity matrix where
    # each idx (a, b) is the dot prod of src embedding a with tgt embedding b.
    dot_prod = embeddings_src * embeddings_tgt

    # For each src word compute softmax activations on similarities with all
    # tgt words (and vice versa).
    # hint: use tf.nn.softmax and vary the axis parameter depending on direction
    softmax_srctgt = tf.nn.softmax(logits= embeddings_src, axis=0 )
    softmax_tgtsrc = tf.nn.softmax(logits=embeddings_tgt, axis=-1)

    # Src->tgt and tgt->src similarities which exceed threshold in both
    # directions are considered to be aligned.
    aligned_subwords = None

    aligned_words = set()
    # we consider any words where at least one subword is aligned to be aligned
    for i, j in zip(*aligned_subwords):
        aligned_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    return words_src, words_tgt, aligned_words

if __name__ == "__main__":
    config = BertConfig.from_pretrained('bert-base-multilingual-cased', output_hidden_states=True)
    model = TFBertModel.from_pretrained('bert-base-multilingual-cased', config=config)
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    src = "BERT is a transformer-based machine learning technique for natural language processing pre-training developed by Google."
    tgt = "BERT ist ein von Google entwickeltes auf Transformern basiertes maschinelles Lernverfahren für das Pre-Training in der natürlichen Sprachverarbeitung."

    words_src, words_tgt, aligned_words = align(src, tgt, model, tokenizer)

    for src, tgt in aligned_words:
        print("Aligned:", words_src[src], words_tgt[tgt])
