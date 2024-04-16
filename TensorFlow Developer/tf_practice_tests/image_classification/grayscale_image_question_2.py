# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset which includes black-and-white images of 10 digits (0-9). The input shape
# should be (28, 28, 1)

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

import tensorflow as tf
import tensorflow_datasets as tfds
from keras import Sequential
from keras.saving import load_model


# Use Tensorflow datasets
def my_model():
    # Load the MNIST dataset
    (train_ds, test_ds), info = tfds.load(name='mnist', split=['train', 'test'], with_info=True, as_supervised=True)

    # Preprocess the training data
    train_ds = train_ds.map(normalize).cache().shuffle(info.splits['train'].num_examples).batch(32)

    # Preprocess the test data
    test_ds = test_ds.map(normalize).cache().batch(32)

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Define the early stopping callback
    # ADD CODE HERE

    # Train the model
    # ADD CODE HERE

    return model


# A function to normalize the data
def normalize(image, label):
    return tf.cast(x=image, dtype=tf.float32) / 255.0, label


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = my_model()
    filepath = "grayscale_model_2.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
