import tensorflow as tf
from config import *
from utils.file_utils import *
from utils.log_generator import *
from utils.utils import *


def get_predictions(model, audio_data):
    predictions = np.array(model.predict(audio_data))
    prediction_value = predictions.argmax()
    # norm_prediction = normalize_gray(predictions)*100
    predictions_array = np.array(predictions[0])
    predictions_sum = predictions_array.sum()
    return prediction_value, predictions_array, predictions_sum


def iterate_test_files(model):
    for file_name in sorted(glob.glob(test_rec_folder + "*.wav")):
        audio_data = get_wav_data(file_name)
        prediction_value, predictions_array, predictions_sum = get_predictions(model, audio_data)
        print("Test file Name: ", file_name, " The Model Predicts:", prediction_value)
        for digit in range(digits_count):
            confidence = np.round((predictions_array[digit] / predictions_sum) * 100)
            print("Class ", digit, " Confidence: ", confidence)
        # print("TestFile Name: ",file_name, " Values:", predictions)
        print("_____________________________\n")


def evaluate_results(x_test, y_test, trained_model, x_labels):
    print("Result evaluating")
    opt = keras.optimizers.adam(lr=0.0001)
    trained_model.compile(loss='categorical_crossentropy',
                          optimizer=opt,
                          metrics=['accuracy'])
    scores = trained_model.evaluate(x_test, y_test, batch_size=32, verbose=1)
    print(scores)
    print("Test Accuracy", scores[1] * 100)
    print("Loss value", scores[0])
    try:
        plot_losses(trained_model)
    except:
        print("error")
    y_pred = trained_model.predict_classes(x_test)
    # print(y_pred)
    # x_labels = [str(i) for i in range(0, 200)]
    # print(x_labels)
    confusion_matrix = tf.math.confusion_matrix(labels=y_pred, predictions=y_pred, name='confusion_matrix')
    with tf.Session():
        print('Confusion Matrix: \n\n', tf.Tensor.eval(confusion_matrix, feed_dict=None, session=None))
    # print(confusion_matrix)
