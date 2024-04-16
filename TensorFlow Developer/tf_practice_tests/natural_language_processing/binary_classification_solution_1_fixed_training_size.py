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
from keras.layers import Embedding, Dense
from keras.src.callbacks import EarlyStopping
from keras.src.layers import GlobalAveragePooling1D
from keras.src.models import load_model
from keras.src.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences


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
    training_size = 400

    # Split the dataset into training and validation sets
    training_sentences = texts[0:training_size]
    testing_sentences = texts[training_size:]
    training_labels = labels[0:training_size]
    validation_labels = labels[training_size:]

    # Tokenize the texts
    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)
    tokenizer.fit_on_texts(training_sentences)
    training_sequences = tokenizer.texts_to_sequences(training_sentences)
    testing_sequences = tokenizer.texts_to_sequences(testing_sentences)

    # Pad the sequences
    padded_training_set = pad_sequences(sequences=training_sequences,
                                        maxlen=max_length,
                                        padding=padding_type,
                                        truncating=trunc_type)
    padded_validation_set = pad_sequences(sequences=testing_sequences,
                                          maxlen=max_length,
                                          padding=padding_type,
                                          truncating=trunc_type)

    # Convert the labels to numpy array
    training_labels = np.array(training_labels)
    validation_labels = np.array(validation_labels)

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
        Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=max_length),
        GlobalAveragePooling1D(),
        Dense(units=1, activation='sigmoid')
    ])

    # Compile the model
    # ADD CODE HERE
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Define an early stopping callback
    # ADD CODE HERE
    early_stop = EarlyStopping(monitor='val_loss', patience=5)

    # Train the model
    # ADD CODE HERE
    model.fit(x=padded_training_set,
              y=training_labels,
              epochs=50,
              validation_data=(padded_validation_set, validation_labels),
              callbacks=[early_stop])

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
