#!/usr/bin/env python
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer
from tqdm import tqdm

def read_dataset(filename, model_name="bert-base-uncased"):
    """Reads a dataset from the specified path and returns sentences and labels"""

    tokenizer = BertTokenizer.from_pretrained(model_name)
    with open(filename, "r") as f:
        lines = f.readlines()
        # preallocate memory for the data
        sents, labels = list(), np.empty((len(lines), 1), dtype=int)

        for i, line in enumerate(lines):
            text, str_label, _ = line.split("\t")
            labels[i] = int(str_label.split("=")[1] == "POS")
            sents.append(text)
    return dict(tokenizer(sents, padding=True, truncation=True, return_tensors="tf")), labels


class BertMLP(tf.keras.Model):
    def __init__(self, embed_batch_size=100, model_name="bert-base-cased"):
        super(BertMLP, self).__init__()
        self.bs = embed_batch_size
        self.model = TFBertModel.from_pretrained(model_name)
        self.classification_head = tf.keras.models.Sequential(
            layers = [
                tf.keras.Input(shape=(self.model.config.hidden_size,)),
                tf.keras.layers.Dense(350, activation="tanh"),
                tf.keras.layers.Dense(200, activation="tanh"),
                tf.keras.layers.Dense(50, activation="tanh"),
                tf.keras.layers.Dense(1, activation="sigmoid", use_bias=False)
            ]
        )

    #@tf.function
    def call(self, inputs):
        # YOUR CODE HERE
        classifier_model = self.classification_head(tf.constant(inputs))
        raise NotImplementedError

def evaluate(model, inputs, labels, loss_func):
    mean_loss = tf.keras.metrics.Mean(name="train_loss")
    accuracy = tf.keras.metrics.BinaryAccuracy(name="train_accuracy")

    predictions = model(inputs)
    mean_loss(loss_func(labels, predictions))
    accuracy(labels, predictions)

    return mean_loss.result(), accuracy.result() * 100


if __name__ == "__main__":
    train = read_dataset("datasets/rt-polarity.train.vecs")
    dev = read_dataset("datasets/rt-polarity.dev.vecs")
    test = read_dataset("datasets/rt-polarity.test.vecs")

    mlp = BertMLP()
    mlp.compile(tf.keras.optimizers.SGD(learning_rate=0.01), loss='mse')
    dev_loss, dev_acc = evaluate(mlp, *dev, tf.keras.losses.MeanSquaredError())
    print("Before training:", f"Dev Loss: {dev_loss}, Dev Acc: {dev_acc}")
    mlp.fit(*train, epochs=10, batch_size=10)
    dev_loss, dev_acc = evaluate(mlp, *dev, tf.keras.losses.MeanSquaredError())
    print("After training:", f"Dev Loss: {dev_loss}, Dev Acc: {dev_acc}")
