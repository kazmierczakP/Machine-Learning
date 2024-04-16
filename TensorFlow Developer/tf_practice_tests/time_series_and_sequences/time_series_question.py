# Copyright (c) 2023, Troy Phat Tran (Mr. Troy).

# Question:

# Build and train a Sequential model that can predict the level of humidity for 5 cities over the time using the
# cities-humidity.csv dataset. The normalized dataset should have a mean absolute error (MAE) of 0.15 or less.

# Your task is to fill in the missing parts of the code block (where commented as "ADD CODE HERE").

# Specific requirements:

# 1. Input shape: (batch_size = 8, n_past = 6, n_features = 5)
#    n_past means a window of the past 6 observations
#    n_features means 5 features (cities) to predict

# 2. Output shape: (batch_size = 8, n_future = 6, n_features = 5)
#    n_future means the next 6 observations to predict


import os
from urllib.request import urlretrieve

import pandas as pd
from keras import Sequential
from keras.callbacks import Callback
from keras.src.saving.saving_api import load_model
from tensorflow.python.data import Dataset
from tensorflow.python.framework.random_seed import set_seed


def time_series_model():
    # Download the dataset
    csv_file = 'cities-humidity.csv'
    if not os.path.exists(csv_file):
        url = 'https://trientran.github.io/tf-practice-exams/cities-humidity.csv'
        urlretrieve(url=url, filename=csv_file)

    # Read the CSV
    df = pd.read_csv(csv_file, sep=',', index_col='date', header=0)

    # Normalize the data
    data = df.values
    data = data - data.min(axis=0)
    data = data / data.max(axis=0)

    # Define a variable to hold the number of features/cities in the dataset.
    n_features = len(df.columns)

    # Some default constants (feel free to update these if the real exam provides different ones)
    n_past = 6
    n_future = 6
    batch_size = 8

    # Set seed to persist training results
    set_seed(1)

    # Split into training and validation sets.
    split_time = int(len(data) * 0.5)
    x_train = data[:split_time]
    x_valid = data[split_time:]

    # Create windowed train and validation sets
    train_set = windowed_dataset(series=x_train, batch_size=batch_size, n_past=n_past, n_future=n_future)
    valid_set = windowed_dataset(series=x_valid, batch_size=batch_size, n_past=n_past, n_future=n_future)

    # Define your model
    model = Sequential([
        # ADD CODE HERE
    ])

    # Compile the model
    # ADD CODE HERE

    # Optional: Print out summary log to double-check input and output shapes
    # ADD CODE HERE (optional)

    # Optional: Define callbacks
    # ADD CODE HERE (optional)

    # Trains the model
    # ADD CODE HERE

    return model


class MyCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}
        val_mae = logs.get('val_mae')
        if val_mae <= 0.15:  # Very importantly, you must change this number if the test expects a certain limit of MAE.
            # For example, this test requires an MAE of 0.15 or less. So it makes sense to set this number to 0.15
            print(f"\nReached {val_mae} Mean Absolute Error after {epoch} epochs so stopping training!")
            self.model.stop_training = True


# A function to create windowed dataset. Derived from
# https://colab.research.google.com/github/lmoroney/dlaicourse/blob/master/TensorFlow%20In%20Practice/Course%204%20-%20S%2BP/S%2BP%20Week%204%20Exercise%20Answer.ipynb
def windowed_dataset(series, batch_size, n_past, n_future):
    ds = Dataset.from_tensor_slices(series)
    ds = ds.window(size=n_past + n_future, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda w: w.batch(n_past + n_future))
    ds = ds.map(lambda w: (w[:n_past], w[n_past:]))
    return ds.batch(batch_size).prefetch(1)


# ===============DO NOT EDIT THE BELOW================================
# Train and save the model
if __name__ == '__main__':
    # Run and save your model
    my_model = time_series_model()
    filepath = "time_series_model.h5"
    my_model.save(filepath)

    # Reload the saved model
    saved_model = load_model(filepath)

    # Show the model architecture
    saved_model.summary()
