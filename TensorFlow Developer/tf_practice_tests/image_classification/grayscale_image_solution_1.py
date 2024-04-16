# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).
# Yann LeCun and Corinna Cortes hold the copyright of MNIST dataset, which is a derivative work from original NIST
# datasets. MNIST dataset is made available under the terms of the Creative Commons Attribution-Share Alike 3.0 license.

# Question:

# Create a classifier for the MNIST dataset which includes black-and-white images of 10 digits (0-9). Link:
# https://www.tensorflow.org/datasets/catalog/mnist.
# The input shape should be (28, 28, 1) because each image has 28*28 pixels and is grayscale.

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.saving import load_model


# Use Keras dataset
def my_model():
    # Load the MNIST dataset
    dataset = mnist
    (x_train, y_train), (x_valid, y_valid) = dataset.load_data()

    # Normalize the input data
    x_train = x_train.astype('float32') / 255.0
    x_valid = x_valid.astype('float32') / 255.0

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        # black-white images have only one color channel
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(units=10, activation='softmax')  # units=10 because there are 10 classes/digits in the dataset
    ])

    # Compile the model
    # ADD CODE HERE
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # Define the early stopping callback
    # ADD CODE HERE
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, min_delta=0.01, verbose=1)

    # Train the model with the early stopping callback
    # Reshape the data to add the channel dimension
    # ADD CODE HERE
    x_train = x_train.reshape((-1, 28, 28, 1))
    x_valid = x_valid.reshape((-1, 28, 28, 1))
    model.fit(
        x=x_train,
        y=y_train,
        epochs=10,
        validation_data=(x_valid, y_valid),
        callbacks=[early_stop]
    )

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
