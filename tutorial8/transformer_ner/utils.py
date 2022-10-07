from functools import partial
import numpy as np
import tensorflow as tf


def sample_generator(dataset, tokenizer, shuffle, pad_to_multiple_of=None):
    # Trim off the last partial batch if present
    if shuffle:
        sample_ordering = np.random.permutation(len(dataset))
    else:
        sample_ordering = np.arange(len(dataset))
    for sample_idx in sample_ordering:
        example = dataset[int(sample_idx)]
        # Handle dicts with proper padding and conversion to tensor.
        example = tokenizer.pad(
            example, return_tensors="np", pad_to_multiple_of=pad_to_multiple_of
        )
        if tokenizer.pad_token_id is not None:
            example["labels"][example["attention_mask"] == 0] = -100
        example = {key: tf.convert_to_tensor(arr) for key, arr in example.items()}

        yield example, example[
            "labels"
        ]  # TF needs some kind of labels, even if we don't use them
    return


def dataset_to_tf(dataset, tokenizer, total_batch_size, num_epochs, shuffle):
    train_generator = partial(sample_generator, dataset, tokenizer, shuffle=shuffle)
    train_signature = {
        feature: tf.TensorSpec(shape=(None,), dtype=tf.int64)
        for feature in dataset.features
        if feature != "special_tokens_mask"
    }
    # This may need to be changed depending on your particular model or tokenizer!
    padding_values = {
        key: tf.convert_to_tensor(0, dtype=tf.int64) for key in dataset.features
    }
    padding_values["labels"] = tf.convert_to_tensor(-100, dtype=tf.int64)
    if tokenizer.pad_token_id is not None:
        padding_values["input_ids"] = tf.convert_to_tensor(
            tokenizer.pad_token_id, dtype=tf.int64
        )
    train_signature["labels"] = train_signature["input_ids"]
    train_signature = (train_signature, train_signature["labels"])
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
    tf_dataset = (
        tf.data.Dataset.from_generator(
            train_generator, output_signature=train_signature
        )
        .with_options(options)
        .padded_batch(
            batch_size=total_batch_size,
            drop_remainder=True,
            padding_values=(padding_values, np.array(0, dtype=np.int64)),
        )
        .repeat(int(num_epochs))
    )
    return tf_dataset
