import tensorflow as tf
from config import *
from file_adapters.files_adapter import *
from model_services.log_generator import *
from utilities.array_processor import *


def get_predictions(model, audio_data):
    """
    Get prediction value for the audio file.
    """
    predictions = np.array(model.predict(audio_data))
    prediction_value = predictions.argmax()
    # norm_prediction = normalize_gray(predictions)*100
    predictions_array = np.array(predictions[0])
    predictions_sum = predictions_array.sum()
    return prediction_value, predictions_array, predictions_sum


def iterate_test_files(model):
    """
    Iterate test files from the test_record folder.
    """
    for file_name in sorted(glob.glob(test_rec_folder + "*.wav")):  # get all .wav files
        audio_data = get_wav_data(file_name)
        prediction_value, predictions_array, predictions_sum = get_predictions(model,
                                                                               audio_data)  # get prediction for the audio file
        print("Test file Name: ", file_name, " The Model Predicts:", prediction_value)
        for digit in range(digits_count):
            confidence = np.round((predictions_array[digit] / predictions_sum) * 100)  # find a confidance
            print("Class ", digit, " Confidence: ", confidence)
        # print("TestFile Name: ",file_name, " Values:", predictions)
        print("_____________________________\n")


def evaluate_results(history, x_test, y_test, trained_model, x_labels):
    """
    evaluate results of the model training.

    """
    print("Result evaluating")
    opt = keras.optimizers.adam(lr=0.0001)  # set optimization value
    trained_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])  # compile file with accuracy and loss metrics
    scores = trained_model.evaluate(x_test, y_test, batch_size=32, verbose=1)  # evaluate a model
    print(scores)
    print("Test Accuracy", scores[1] * 100)
    print("Loss value", scores[0])
    # plot_losses(history)
    y_pred = trained_model.predict_classes(x_test)  # get predicted classes value
    confusion_matrix = tf.math.confusion_matrix(labels=y_pred, predictions=y_pred,
                                                name='confusion_matrix')  # create a result confusion matrix
    with tf.Session():
        print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None))
