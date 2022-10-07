#!/usr/bin/env python
import random

import numpy as np
import tensorflow as tf
from datasets import load_dataset, load_metric, Dataset

from transformers import (
    AutoConfig,
    AutoTokenizer,
    TFAutoModelForTokenClassification,
    TFTrainingArguments,
    create_optimizer,
)

from utils import dataset_to_tf


model_name = "distilbert-base-uncased"
output_dir = "./models"
dataset = "conll2003"
max_train_samples = 3000
max_eval_samples = 250


def main():
    raw_datasets = load_dataset(dataset)
    features = raw_datasets["train"].features
    text_column_name = "tokens"
    label_column_name = "ner_tags"

    label_list = features[label_column_name].feature.names
    label_to_id = {i: i for i in range(len(label_list))}
    num_labels = len(label_list)

    config = AutoConfig.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    # Tokenize all texts and align the labels with them.
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples[text_column_name],
            truncation=True,
            # We use this argument because the texts in our dataset are lists of words (with a label for each word).
            is_split_into_words=True,
        )

        labels = []
        for i, label in enumerate(examples[label_column_name]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label_to_id[label[word_idx]])
                # For the other tokens in a word, we set the label to -100
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            labels.append(label_ids)
        tokenized_inputs["labels"] = labels
        return tokenized_inputs

    processed_raw_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
        desc="Running tokenizer on dataset",
    )

    train_dataset = Dataset.from_dict(
        processed_raw_datasets["train"][:max_train_samples]
    )
    eval_dataset = Dataset.from_dict(
        processed_raw_datasets["validation"][:max_eval_samples]
    )

    # Print a few random samples from the training set:
    for index in random.sample(range(len(train_dataset)), 3):
        print(f"Sample {index} of the training set: {train_dataset[index]}.")

    training_args = TFTrainingArguments(output_dir=output_dir, num_train_epochs=1)
    with training_args.strategy.scope():
        # Initialize model
        model = TFAutoModelForTokenClassification.from_pretrained(
            model_name, config=config
        )

        # Create TF datasets
        num_replicas = training_args.strategy.num_replicas_in_sync
        total_train_batch_size = (
            training_args.per_device_train_batch_size * num_replicas
        )
        train_batches_per_epoch = len(train_dataset) // total_train_batch_size
        tf_train_dataset = dataset_to_tf(
            train_dataset,
            tokenizer,
            total_batch_size=total_train_batch_size,
            num_epochs=training_args.num_train_epochs,
            shuffle=True,
        )
        total_eval_batch_size = training_args.per_device_eval_batch_size * num_replicas
        eval_batches_per_epoch = len(eval_dataset) // total_eval_batch_size
        tf_eval_dataset = dataset_to_tf(
            eval_dataset,
            tokenizer,
            total_batch_size=total_eval_batch_size,
            num_epochs=training_args.num_train_epochs,
            shuffle=False,
        )

        # Optimizer, loss and compilation
        optimizer, _ = create_optimizer(
            init_lr=training_args.learning_rate,
            num_train_steps=int(
                training_args.num_train_epochs * train_batches_per_epoch
            ),
            num_warmup_steps=training_args.warmup_steps,
            adam_beta1=training_args.adam_beta1,
            adam_beta2=training_args.adam_beta2,
            adam_epsilon=training_args.adam_epsilon,
            weight_decay_rate=training_args.weight_decay,
        )

        def dummy_loss(_, y_pred):
            return tf.reduce_mean(y_pred)

        model.compile(loss={"loss": dummy_loss}, optimizer=optimizer)
        model.fit(tf_train_dataset, validation_data=tf_eval_dataset, epochs=int(training_args.num_train_epochs),steps_per_epoch=train_batches_per_epoch, validation_steps=eval_batches_per_epoch)



        # Metrics
        metric = load_metric("seqeval")

        def get_labels(y_pred, y_true):
            # Transform predictions and references tensors to numpy arrays

            # Remove ignored index (special tokens)
            true_predictions = [
                [label_list[p] for (p, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            true_labels = [
                [label_list[l] for (_, l) in zip(pred, gold_label) if l != -100]
                for pred, gold_label in zip(y_pred, y_true)
            ]
            return true_predictions, true_labels

        def compute_metrics():
            results = metric.compute()
            return {
                "precision": results["overall_precision"],
                "recall": results["overall_recall"],
                "f1": results["overall_f1"],
                "accuracy": results["overall_accuracy"],
            }

        eval_inputs = {
            key: tf.ragged.constant(eval_dataset[key]).to_tensor()
            for key in eval_dataset.features
        }
        predictions = model.predict(
            eval_inputs, batch_size=training_args.per_device_eval_batch_size
        )["logits"]
        predictions = tf.math.argmax(predictions, axis=-1)
        labels = np.array(eval_inputs["labels"])
        labels[np.array(eval_inputs["attention_mask"]) == 0] = -100
        preds, refs = get_labels(predictions, labels)
        metric.add_batch(
            predictions=preds,
            references=refs,
        )
        eval_metric = compute_metrics()
        print("Evaluation metrics:")
        for key, val in eval_metric.items():
            print(f"{key}: {val:.4f}")

    # Save model
    model.save_pretrained(output_dir)


if __name__ == "__main__":
    main()
