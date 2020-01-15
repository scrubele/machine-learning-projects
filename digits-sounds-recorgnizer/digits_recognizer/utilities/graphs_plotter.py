from keras import Model
import matplotlib.pyplot as plt
from config import *
from file_adapters.files_adapter import *
import json

def plot_losses(trained_model):
    # Get the dictionary containing each metric and the loss for each epoch
    history_dict = trained_model.history
    # print(trained_model.history)
    json.dump(history_dict, open(HISTORY_JSON, 'w'))
    plt.plot(trained_model.history['accuracy'], label='accuracy')
    plt.plot(trained_model.history['val_accuracy'], label='val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()
    plt.savefig('accuracy.png', dpi=300, frameon='false')


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


def visualize_CNN(model, X_train):
    layer_outputs = [layer.output for layer in model.layers]
    activation_model = Model(inputs=model.input, outputs=layer_outputs)
    activations = activation_model.predict(X_train[10].reshape(1, 28, 28, 1))

    def display_activation(activations, col_size, row_size, act_index):
        activation = activations[act_index]
        activation_index = 0
        fig, ax = plt.subplots(row_size, col_size, figsize=(row_size * 2.5, col_size * 1.5))
        for row in range(0, row_size):
            for col in range(0, col_size):
                ax[row][col].imshow(activation[0, :, :, activation_index], cmap='gray')
                activation_index += 1

    display_activation(activations, 8, 8, 1)
