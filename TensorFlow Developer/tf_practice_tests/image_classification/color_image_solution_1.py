# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).

# Binary (2-classes) image classification dataset: apple-banana.
# Direct link:
# http://dl.dropboxusercontent.com/scl/fi/mw43x41744wfykvm8fljx/apple-banana.zip?rlkey=ghmo9zf2rpza2uq9xbf7jpe3e&dl=0
# (~7.6 Megabytes)
# Backup direct link: https://trientran.github.io/tf-practice-exams/apple-banana.zip

# This dataset comprises 2 classes namely Apple and Banana, and it has been split into training and validation sets.
# Create a classifier for the given dataset. The required input shape must be 100x100x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

import os
import zipfile
from urllib.request import urlretrieve

from keras import Sequential
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.models import load_model
from keras.utils import image_dataset_from_directory
from tensorflow import cast, float32
from tensorflow.python.data import AUTOTUNE


# A function to rescale/normalize images
def rescale(image, label):
    image = cast(image, float32) / 255.0
    return image, label


def binary_model():
    # Define a data folder to extract our compressed dataset to
    data_folder = 'apple-banana/'

    # Download and extract the dataset if not existing
    if not os.path.exists(data_folder):
        dataset_url = 'http://dl.dropboxusercontent.com/scl/fi/mw43x41744wfykvm8fljx/apple-banana.zip?rlkey=ghmo9zf2rpza2uq9xbf7jpe3e&dl=0'
        local_zip = 'apple-banana.zip'
        urlretrieve(url=dataset_url, filename=local_zip)
        zip_ref = zipfile.ZipFile(file=local_zip, mode='r')
        zip_ref.extractall(data_folder)
        zip_ref.close()

    # Define image size and batch size
    img_size = (100, 100)
    batch_size = 32

    # Create the training dataset
    # The dataset is already split into training and validation sets
    train_ds = image_dataset_from_directory(
        directory="apple-banana/train/",
        seed=1,
        image_size=img_size,
        batch_size=batch_size
    )

    # Create the validation dataset
    val_ds = image_dataset_from_directory(
        directory="apple-banana/validation/",
        seed=1,
        image_size=img_size,
        batch_size=batch_size
    )

    # Rescale images (option 1)
    train_ds = train_ds.map(rescale, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.map(rescale, num_parallel_calls=AUTOTUNE)

    # Configure the dataset for performance
    # https://www.tensorflow.org/tutorials/images/classification
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    # Define the model architecture
    model = Sequential([
        # Rescaling(1. / 255, input_shape=(img_size[0], img_size[1], 3)),  # Rescale images (option 2)
        # ADD CODE HERE
        Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(img_size[0], img_size[1], 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(filters=64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(units=64, activation='relu'),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    # ADD CODE HERE
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Define the early stopping callback for val_accuracy
    # ADD CODE HERE
    early_stop = EarlyStopping(monitor='val_accuracy', patience=5, verbose=1, min_delta=0.01)

    # Show the model architecture (optional)
    summarize_model(model)

    # Train the model with early stopping callback
    # ADD CODE HERE
    model.fit(x=train_ds, epochs=5, validation_data=val_ds, callbacks=[early_stop])

    return model


# ===============DO NOT EDIT THIS PART================================
def summarize_model(model):
    model.summary()
    input_shape = model.layers[0].input_shape
    print(f'Input shape: {input_shape}')


if __name__ == '__main__':
    # Run and save your model
    my_model = binary_model()
    filepath = "binary_rgb_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)
    summarize_model(saved_model)
