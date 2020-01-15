from random import random

import contextlib
import glob
import os
import random
import time
import wave

import keras
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from keras.models import Sequential
from keras.models import model_from_json
from scipy.io import wavfile
from skimage.measure import block_reduce
from file_adapters.files_adapter import *
from file_adapters.wav_processor import *
from model_services.log_generator import *
from model_services.results_evaluator import *
from utilities.array_processor import *
from config import *


def form_test_dataset(file_names, file_count):
    """
    Randomly create a test dataset (10 percent files of the recordings folder)
    Args:
        file_names: our files from the recordings folder
        file_count: a number of files in the folder
    """
    test_dataset = []
    speakers_count = int(file_count / examples_per_speaker) # count a number of files per speaker
    for speaker in range(speakers_count):
        random_recordings = random.sample(
            file_names[(speaker * examples_per_speaker + 1):(speaker + 1) * examples_per_speaker],
            int(examples_per_speaker / test_files_percentage))
        test_dataset.extend(random_recordings)
    return test_dataset


def progress_data(audio_dir, data_list, y_list, x_list, data_type):
    """
    Change audio signals into pictures and process it.
    """
    for item, file in enumerate(data_list):
        y_list[item] = int(file[0])
        file_path = audio_dir + file
        spectrogram, gray_gram, normalized_gram, normalized_gram_shape, red_gram = get_grams(file_path) # get color grams of audio signal
        if normalized_gram_shape[0] > 150:
            continue
        x_list[item, :, :] = red_gram # assign red gram to our result list
        print('Progress ' + data_type + ' Data: {0:.2f}%'.format(float(item) * 100 / len(data_list))
              # , end="\r"
              )


def reshape_vectors(x_train, y_train, x_test, y_test):
    """
    Converts a class vector of our lists to binary class matrixes.
    """
    y_train = keras.utils.to_categorical(y_train, digits_count)  # Converts a class int vector to binary class matrix.
    y_test = keras.utils.to_categorical(y_test, digits_count)
    x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
    x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)
    return x_train, y_train, x_test, y_test


def create_datasets(audio_dir):
    """
    Split the dataset into test and train sets randomly.
    Args:
        audio_dir:
    Returns:
        x_train, y_train, x_test, y_test
    """
    file_names, file_count = get_recordings_files_names(audio_dir)
    test_dataset = form_test_dataset(file_names, file_count)  # form our test dataset from recordings (10%)
    train_dataset = [file for file in file_names if
                     file not in test_dataset]  # form our train dataset from recordings (90%)
    x_test, x_train, y_test, y_train = create_empty_arrays(test_dataset, train_dataset)
    progress_data(audio_dir, test_dataset, y_test, x_test, 'Test')  # change audio signals into pictures (test dataset)
    progress_data(audio_dir, train_dataset, y_train, x_train,
                  'Train')  # change audio signals into pictures (train dataset)
    print_datasets_properties(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = reshape_vectors(x_train, y_train, x_test,
                                                       y_test)  # from vectors to matrixes
    return x_train, y_train, x_test, y_test, test_dataset


def add_layers_to_the_model(model, input_shape):
    """
    Add layers in 3 steps (Convolution, Flattering and Full connection)
    Args:
        model:  a model to which we want add layers
        input_shape: shape of spectrogram
    Returns:
        model that can be trained
    """
    # Step 1 - Convolution
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    # Step 2 - Flattening
    model.add(Flatten())
    # Step 3 - Full connection
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(digits_count, activation='softmax'))
    return model


def create_model(path):
    """
    Create a trained model.
    Args:
        a path of recordings folder
    Returns:
        a trained model
    """
    x_train, y_train, x_test, y_test, test_dataset = create_datasets(path)  # create train and test datasets
    model = Sequential()  # create a Sequential (linear) CNN model
    input_shape = (image_height, image_width, 1)  # shape of our spectrograms
    model = add_layers_to_the_model(model, input_shape)  # create layers in the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(), metrics=['accuracy'])  # compile and set loss and metrics
    print(model.summary())  # our model summary
    history = model.fit(x_train, y_train, batch_size=4, epochs=10,
                        verbose=1, validation_data=(x_test, y_test))  # train our model
    print(history.history.keys())
    print(history.history)
    evaluate_results(history.history, x_test, y_test, model, test_dataset)  # evaluate training results
    return model


def load_model():  # load a model
    print(os.path.isfile(MODEL_JSON))
    if os.path.isfile(MODEL_JSON):  # check if model_json exists
        model = load_model_from_disk()  # load an existed model
    else:
        model = create_model(recording_directory)  # create a model in the recording directory
        save_model_to_disk(model)  # save model to the folder
    return model


if __name__ == '__main__':
    if LOG_MODE:  # check if this is logging mode
        generate_logs(recording_directory, 6)  # run generate log function
    else:
        trained_model = load_model()  # load a model
        trained_model_x_train, _, trained_model_x_test, trained_model_y_test, trained_model_test_dataset = \
            create_datasets(recording_directory)  # create testing datasets
        # evaluate results of testing
        evaluate_results(_, trained_model_x_test, trained_model_y_test, trained_model, trained_model_test_dataset)
        iterate_test_files(trained_model)  # test files from the test_record folder
