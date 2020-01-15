import json

import matplotlib.pyplot as plt
from config import *
from file_adapters.files_adapter import *
from keras import Model


def plot_losses(trained_model):
    # Get the dictionary containing each metric and the loss for each epoch
    # history_dict = trained_model.history
    # print(trained_model.keys())
    print(trained_model)
    # print(type(trained_model))
    # history_dict = {index: v for index, v in np.ndenumerate(trained_model)}
    # print(trained_model.history)
    try:
        json.dump(trained_model, open(HISTORY_JSON, 'w'))
    except:
        x = 0
    print(list(trained_model.keys()))
    plt.figure(figsize=(1, 1))
    plt.plot(trained_model['accuracy'], label='accuracy')
    plt.plot(trained_model['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.rcParams["figure.figsize"] = (100, 200)
    plt.savefig('accuracy.png', dpi=300, frameon='false')
    plt.show()


def plot_amplitude_graph(current_logs_image_folder, file_name, file_data):
    figure, axes = plt.subplots(nrows=1, ncols=1)  # create figure & 1 axis
    axes.set_title('Sound of ' + file_name[0] + ' - Sampled audio signal in time')
    axes.set_xlabel('Sample number')
    axes.set_ylabel('Amplitude')
    axes.plot(file_data)  # save the figure to file
    figure.savefig(os.path.join(current_logs_image_folder, file_name[0:5] + '.png'))
    plt.close(figure)


def plot_frequency_graph(current_logs_image_folder, file_name, file_data, file_rate):
    figure, axes = plt.subplots(1)
    # fig.subplots_adjust(left=0,right=1,bottom=0,top=1)
    # ax.axis('off')
    spectrum, frequencies, times, image = axes.specgram(x=file_data, Fs=file_rate, noverlap=511, NFFT=512)
    # ax.axis('off')
    # plt.rcParams['figure.figsize'] = [0.75,0.5]
    color_bar = figure.colorbar(image)
    color_bar.set_label('Intensity dB')
    # ax.axis("tight")
    axes.set_title('Spectrogram of spoken ' + file_name[0])
    axes.set_xlabel('time')
    axes.set_ylabel('frequency Hz')
    figure.savefig(os.path.join(
        current_logs_image_folder, file_name[0] + '_spec.png'), dpi=300, frameon='false')
    plt.close(figure)

