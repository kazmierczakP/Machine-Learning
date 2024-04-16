# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).

# Question:

# Build and train a Sequential model that can predict the level of humidity for My City using the my-city-humidity.csv
# dataset. In this particular problem, we only need to predict the sunspot activity based on the previous values of the
# series not the time steps, so you don't need to include the time step as a feature in the model. The normalized
# dataset should have a mean absolute error (MAE) of 0.15 or less.

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").


import csv
import os

import numpy as np
import tensorflow as tf
from keras import Sequential
from keras.callbacks import Callback
from keras.saving import load_model
from keras.src.utils.data_utils import urlretrieve


def sequences_model():
    # Download the dataset
    csv_file = 'my-city-humidity.csv'
    if not os.path.exists(csv_file):
        url = 'https://trientran.github.io/tf-practice-exams/my-city-humidity.csv'
        urlretrieve(url=url, filename=csv_file)

    humidity = []

    # Read the CSV and append all the records to humidity
    with open(csv_file) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        next(reader)
        for row in reader:
            humidity.append(float(row[2]))

    series = np.array(humidity)

    # Normalize the data
    min_value = np.min(series)
    max_value = np.max(series)
    series -= min_value
    series /= max_value

    # The data is split into training and validation sets at time step 2900 (~90% of the number of records). When it
    # comes to the real test, the dataset may be bigger or smaller than this dataset. They may have already set this
    # value for you or you must calculate it yourself.
    split_step = 2900

    # In this particular problem, we only need to predict the sunspot activity based on the previous values of the
    # series, so we don't need to include the time step as a feature in the model. Therefore, we only use the x_train
    # and x_valid variables (not time_train nor time_valid), which contain the normalized sunspot activity values for
    # the training and validation sets.
    x_train = series[:split_step]
    x_valid = series[split_step:]

    # Some default constants
    shuffle_buffer = 1000
    batch_size = 32
    window_size = 30

    train_set = windowed_dataset(
        series=x_train,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer
    )

    valid_set = windowed_dataset(
        series=x_valid,
        window_size=window_size,
        batch_size=batch_size,
        shuffle_buffer=shuffle_buffer
    )

    # Define your model
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Optional: Define early stopping callbacks.
    # ADD CODE HERE

    # Train the model
    # ADD CODE HERE

    return model


# If you are aiming at achieving a certain limit of Mean Absolute Error, this callback class will be handy.
class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_mae = logs.get('val_mae')
        # Very importantly, you must change this number if the test expects a certain limit of MAE.For example, this
        # test requires an MAE of 0.15 or less. So it makes sense to set this number to 0.15
        if val_mae <= 0.15:
            print(f"\nReached {val_mae} Mean Absolute Error after {epoch} epochs so stopping training!")
            self.model.stop_training = True


# This code snippet is copied from
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb
def windowed_dataset(series, window_size, batch_size, shuffle_buffer):
    series = tf.expand_dims(series, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(series)
    ds = ds.window(size=window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda w: (w[:-1], w[1:]))
    return ds.batch(batch_size).prefetch(1)


# ===============DO NOT EDIT THE BELOW================================
if __name__ == '__main__':
    # Run and save your model
    my_model = sequences_model()
    filepath = "sequences_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
