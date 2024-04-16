# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).

# Question:

# Build and train a binary classifier for the language classification dataset. The dataset is typically a JSON array
# of 500 JSON objects. Each object has 3 keys: sentence, language_code, and is_english.
# We want our model to be able to determine whether a piece of text is "English or not".

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

# Note: the dataset is imbalanced as there are more non-English sentences than English ones. To keep things simple, 
# you don't need to handle data imbalance in this coding challenge.

import json
import os
from urllib.request import urlretrieve

import numpy as np
from keras import Sequential
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences


def nlp_binary_model():
    # Download the dataset
    json_file = 'language-classification.json'
    if not os.path.exists(json_file):
        url = 'https://trientran.github.io/tf-practice-exams/language-classification.json'
        urlretrieve(url=url, filename=json_file)

    # Parse the JSON file
    with open(file=json_file, mode='r', encoding='utf-8') as f:
        datastore = json.load(f)

    # Extract texts and labels from JSON data
    texts = []
    labels = []
    for item in datastore:
        texts.append(item['sentence'])  # replace with the sentence/paragraph/text field in the real test JSON file
        labels.append(item['is_english'])  # replace with the label field in the real test JSON file

    # Predefined constants
    max_length = 25
    trunc_type = 'pre'  # Can be replaced with 'post'
    vocab_size = 500
    padding_type = 'pre'  # Can be replaced with 'post'
    embedding_dim = 32
    oov_tok = "<OOV>"

    # Split the dataset into training and validation sets
    num_samples = len(texts)
    num_train_samples = int(0.8 * num_samples)
    indices = np.random.permutation(num_samples)
    train_indices = indices[:num_train_samples]
    val_indices = indices[num_train_samples:]

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    # Pad the sequences
    padded_sequences = pad_sequences(
        sequences=sequences,
        maxlen=max_length,
        padding=padding_type,
        truncating=trunc_type
    )
    padded_training_set = padded_sequences[train_indices]
    padded_validation_set = padded_sequences[val_indices]

    # Convert the labels to numpy array
    training_labels = np.array(labels)[train_indices]
    validation_labels = np.array(labels)[val_indices]

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Define an early stopping callback (optional)
    # ADD CODE HERE

    # Train the model
    # ADD CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = nlp_binary_model()
    filepath = "nlp_binary_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
