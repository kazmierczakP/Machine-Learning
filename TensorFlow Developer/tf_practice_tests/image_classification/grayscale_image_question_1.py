# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset which includes black-and-white images of 10 digits (0-9). Link:
# https://www.tensorflow.org/datasets/catalog/mnist.
# The input shape should be (28, 28, 1) because each image has 28*28 pixels and is grayscale.

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

from keras import Sequential
from keras.datasets import mnist
from keras.saving import load_model


# Use Keras dataset
def my_model():
    # Load the MNIST dataset
    dataset = mnist
    (x_train, y_train), (x_test, y_test) = dataset.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Define the early stopping callback
    # ADD CODE HERE

    # Train the model with the early stopping callback
    # ADD CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
if __name__ == '__main__':
    # Run and save your model
    my_model = my_model()
    filepath = "grayscale_model_1.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
