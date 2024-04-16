# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).
# The Malaria dataset available on the official National Institutes of Health (NIH) website is in the public domain and
# does not have any specific license nor copyright restrictions.

# Binary (2-classes) image classification
# Dataset: Malaria.
# Direct link: https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip (~350 Megabytes)

# This dataset comprises 2 classes namely Parasitized and Uninfected, and it is not split into training and test sets
# yet. The images' resolutions are varied.
# Create a classifier for the given dataset. The required input shape must be 40x40x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

import os
import zipfile
from urllib.request import urlretrieve

from keras import Sequential
from keras.models import load_model
from keras.utils import image_dataset_from_directory


def binary_model():
    # Define a data folder to extract our compressed dataset to
    data_folder = 'cell_images/'

    # Download and extract the dataset if not existing
    if not os.path.exists(data_folder):
        dataset_url = 'https://data.lhncbc.nlm.nih.gov/public/Malaria/cell_images.zip'
        local_zip = 'cell_images.zip'
        urlretrieve(url=dataset_url, filename=local_zip)
        zip_ref = zipfile.ZipFile(file=local_zip, mode='r')
        zip_ref.extractall()
        zip_ref.close()

    # Define image size and batch size
    img_size = (0, 0)  # ADD CODE HERE: just update the image size here to match the requirement
    batch_size = 32

    # Create the training dataset. The dataset is not split into training and validation sets yet
    train_ds = image_dataset_from_directory(
        directory=data_folder,
        validation_split=0.2,
        subset='training',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Create the validation dataset
    val_ds = image_dataset_from_directory(
        directory=data_folder,
        validation_split=0.2,
        subset='validation',
        seed=42,
        image_size=img_size,
        batch_size=batch_size
    )

    # Define the model architecture
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Define the early stopping callback for val_accuracy
    # ADD CODE HERE

    # Show the model architecture (optional)
    summarize_model(model)

    # Train the model with early stopping callback
    # ADD CODE HERE

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
