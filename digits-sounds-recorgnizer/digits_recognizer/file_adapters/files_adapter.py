from digits_sounds_recognizer import *


def save_model_to_disk(model):
    """
    save created model
    Args:
        model:

    Returns:

    """
    model_json = model.to_json()  # serialize model to JSON
    with open("%s" % MODEL_JSON, "w") as model_json_file:
        model_json_file.write(model_json)
    model.save_weights("model.h5")  # serialize weights to HDF5
    print("Saved model to disk")


def load_model_from_disk():
    """
    # load json and create model
    Returns:

    """
    with open("%s" % MODEL_JSON, 'r') as model_json_file:
        loaded_model_json = model_json_file.read()
        model_json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights("model.h5")  # load weights into new model
    print("Loaded model from disk")
    return loaded_model


def get_recordings_files_names(audio_dir):
    file_names = [f for f in os.listdir(audio_dir) if '.wav' in f]
    file_names.sort()
    return file_names, len(file_names)