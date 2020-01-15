import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
image_width = 25
image_height = 17
total_examples = 2000
speakers = 4
examples_per_speaker = 50
test_files_percentage = 10
digits_count = 10
slash = '/'
test_rec_folder = "../recordings" + slash + "test_records" + slash
log_image_folder = "../logs" + slash + "log_images" + slash
images_folder = "../results" + slash + "images" + slash
recording_directory = "../recordings" + slash + "all_recordings" + slash
HISTORY_JSON = "../history.json"
MODEL_JSON = "model.json"
LOG_MODE = 0  # 1 for time, 2 for frequency
