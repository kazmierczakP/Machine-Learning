# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).

# Question:

# Multiclass image classification
# Dataset: Mr Troy Fruits.
# Direct link: http://dl.dropboxusercontent.com/s/a32yc71tgfvfvku/mr-troy-fruits.zip (~11 Megabytes)
# Back-up direct link: https://trientran.github.io/tf-practice-exams/mr-troy-fruits.zip
# This dataset comprises 3 classes (Banana, Orange, and Apple), and it is not split into training and test sets yet.
# Create a classifier for the given dataset. The required input shape must be 40x40x3 (RGB images).

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").


import os
import zipfile
from urllib.request import urlretrieve

from keras import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.saving import load_model


def multiclass_model():
    # Define a data folder to extract our compressed dataset to
    data_folder = "mr-troy-fruits/"

    # Download and extract the dataset if not existing
    if not os.path.exists(data_folder):
        dataset_url = 'http://dl.dropboxusercontent.com/s/a32yc71tgfvfvku/mr-troy-fruits.zip'
        local_zip = 'mr-troy-fruits.zip'
        urlretrieve(url=dataset_url, filename=local_zip)
        zip_ref = zipfile.ZipFile(file=local_zip, mode='r')
        zip_ref.extractall()
        zip_ref.close()

    # Define image data generator with data augmentation
    training_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.2
    )

    # Constants for image size and batch size
    img_size = (0, 0)  # ADD CODE HERE: just update the image size here to match the requirement
    batch_size = 32

    # Training set
    train_generator = training_datagen.flow_from_directory(
        directory=data_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training'
    )

    # Validation set
    validation_generator = training_datagen.flow_from_directory(
        directory=data_folder,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation'
    )

    # Define model architecture
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Define an early stopping callback
    # ADD CODE HERE

    # Show the model architecture (optional)
    summarize_model(model)

    # Start training
    # ADD CODE HERE

    return model


# ===============DO NOT EDIT THIS PART================================
def summarize_model(model):
    model.summary()
    input_shape = model.layers[0].input_shape
    print(f'Input shape: {input_shape}')


if __name__ == '__main__':
    # Run and save your model
    my_model = multiclass_model()
    filepath = "multiclass_rgb_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)
    summarize_model(saved_model)
