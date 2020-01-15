import numpy as np

from config import *


def print_datasets_properties(x_train, y_train, x_test, y_test):
    print("Size of Training Data:", np.shape(x_train))
    print("Size of Training Labels:", np.shape(y_train))
    print("Size of Test Data:", np.shape(x_test))
    print("Size of Test Labels:", np.shape(y_test))


def create_empty_arrays(test_dataset, train_dataset):
    y_test = np.zeros(len(test_dataset))
    y_train = np.zeros(len(train_dataset))
    x_train = np.zeros((len(train_dataset), image_height, image_width))
    x_test = np.zeros((len(test_dataset), image_height, image_width))
    return x_test, x_train, y_test, y_train
