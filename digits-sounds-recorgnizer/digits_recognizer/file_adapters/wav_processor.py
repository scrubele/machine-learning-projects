import matplotlib.pyplot as plt
import numpy as np
from digits_sounds_recognizer import *
from utilities.graphs_plotter import *


def graph_spectrogram(wav_file_path, nfft=512, noverlap=511):
    """
    Create a spectogram (plot) of existed file.
    Args:
        wav_file_path: audio file path
    """
    # find_duration(wav_file_path)
    rate, data = wavfile.read(wav_file_path)  # read a file
    figure, axes = plt.subplots(1)
    figure.subplots_adjust(left=0, right=1, bottom=0, top=1)
    axes.axis('off')
    spectrum, frequencies, times, image = axes.specgram(x=data, Fs=rate, noverlap=noverlap,
                                                        NFFT=nfft)  # create a spectogram
    axes.axis('off')
    plt.rcParams['figure.figsize'] = [0.75, 0.5]
    # figure.savefig(images_folder + wav_file_path.split(slash)[-1:][0] + '.png', dpi=300, frameon='false')
    # if you need to save figures uncomment line above
    figure.canvas.draw()
    width, height = figure.get_size_inches() * figure.get_dpi() # find width and height of the spectogram
    figure_canvas_string = figure.canvas.tostring_rgb() # convert to rgb
    canvas_array = np.frombuffer(figure_canvas_string, dtype=np.uint8) # convert a canvas
    image_array = np.reshape(canvas_array, (int(height), int(width), 3)) # reshape an array
    # plt.show()
    plt.close(figure)
    return image_array


def rgb2gray(rgb):
    """
    Convert color image to grayscale
    """
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def normalize_gray(array):
    """
    Normalize Gray colored image
    """
    return (array - array.min()) / (array.max() - array.min())


def get_grams(file_path):
    spectrogram = graph_spectrogram(file_path)
    gray_gram = rgb2gray(spectrogram)
    normalized_gram = normalize_gray(gray_gram)
    normalized_gram_shape = normalized_gram.shape
    red_gram = block_reduce(normalized_gram, block_size=(3, 3), func=np.mean)
    return spectrogram, gray_gram, normalized_gram, normalized_gram_shape, red_gram


def get_wav_data(file_path):
    """
    # Extract wave data from recorded audio
    Args:
        file_path:

    Returns:

    """
    spectrogram, gray_gram, normalized_gram, normalized_gram_shape, red_gram = get_grams(file_path)
    if normalized_gram_shape[0] > 100:
        current_red_gram = block_reduce(normalized_gram, block_size=(26, 26), func=np.mean)
    else:
        current_red_gram = red_gram
    red_gram = current_red_gram[0:image_height, 0:image_width]
    red_data = red_gram.reshape(image_height, image_width, 1)
    recording_data = np.empty((1, image_height, image_width, 1))
    recording_data[0, :, :, :] = red_data
    return recording_data
