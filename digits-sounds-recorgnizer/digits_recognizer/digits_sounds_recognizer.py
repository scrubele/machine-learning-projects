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
    test_dataset = []
    speakers_count = int(file_count / examples_per_speaker)
    for speaker in range(speakers_count):
        random_recordings = random.sample(
            file_names[(speaker * examples_per_speaker + 1):(speaker + 1) * examples_per_speaker],
            int(examples_per_speaker / test_files_percentage))
        test_dataset.extend(random_recordings)
    return test_dataset


def progress_data(audio_dir, data_list, y_list, x_list, data_type):
    for item, file in enumerate(data_list):
        y_list[item] = int(file[0])
        file_path = audio_dir + file
        # print(item, y_list[item], file_path)
        spectrogram, gray_gram, normalized_gram, normalized_gram_shape, red_gram = get_grams(file_path)
        # print(normalized_gram_shape[0])
        if normalized_gram_shape[0] > 150:
            continue
        # print(x_test[i,:,:])
        x_list[item, :, :] = red_gram
        print('Progress ' + data_type + ' Data: {0:.2f}%'.format(float(item) * 100 / len(data_list))
              # , end="\r"
              )


def reshape_vectors(x_train, y_train, x_test, y_test):
    y_train = keras.utils.to_categorical(y_train, digits_count)  # Converts a class int vector to binary class matrix.
    y_test = keras.utils.to_categorical(y_test, digits_count)
    x_train = x_train.reshape(x_train.shape[0], image_height, image_width, 1)
    x_test = x_test.reshape(x_test.shape[0], image_height, image_width, 1)
    return x_train, y_train, x_test, y_test


def create_datasets(audio_dir, step=1):
    """
    Split the dataset into test and train sets randomly
    Args:
        step:
        audio_dir:
    Returns:
        x_train, y_train, x_test, y_test
    """
    file_names, file_count = get_recordings_files_names(audio_dir)
    test_dataset = form_test_dataset(file_names, file_count)
    train_dataset = [file for file in file_names if file not in test_dataset]
    x_test, x_train, y_test, y_train = create_empty_arrays(test_dataset, train_dataset)
    progress_data(audio_dir, test_dataset, y_test, x_test, 'Test')
    progress_data(audio_dir, train_dataset, y_train, x_train, 'Train')
    print_datasets_properties(x_train, y_train, x_test, y_test)
    x_train, y_train, x_test, y_test = reshape_vectors(x_train, y_train, x_test, y_test)
    return x_train, y_train, x_test, y_test, test_dataset


def add_layers_to_the_model(model, input_shape):
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
    x_train, y_train, x_test, y_test, test_dataset = create_datasets(path)
    model = Sequential()
    input_shape = (image_height, image_width, 1)
    model = add_layers_to_the_model(model, input_shape)
    model.compile(loss=keras.losses.categorical_crossentropy,
                  optimizer=keras.optimizers.adam(), metrics=['accuracy'])
    print(model.summary())
    history = model.fit(x_train, y_train, batch_size=4, epochs=10,
                        verbose=1, validation_data=(x_test, y_test))
    print(history.history.keys())
    print(history.history)
    evaluate_results(history.history, x_test, y_test, model, test_dataset)
    return model


def load_model():
    print(os.path.isfile(MODEL_JSON))
    if os.path.isfile(MODEL_JSON):
        model = load_model_from_disk()
    else:
        model = create_model(recording_directory)
        save_model_to_disk(model)
    return model


if __name__ == '__main__':
    if LOG_MODE:
        generate_logs(recording_directory, 6)
    else:
        trained_model = load_model()
        trained_model_x_train, _, trained_model_x_test, trained_model_y_test, trained_model_test_dataset = \
            create_datasets(recording_directory, 100)
        evaluate_results(_, trained_model_x_test, trained_model_y_test, trained_model, trained_model_test_dataset)
        # visualize_CNN(trained_model, trained_model_x_train)
        # iterate_test_files(trained_model)
