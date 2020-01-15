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
test_rec_folder = "test_records" + slash
log_image_folder = "log_images" + slash
images_folder = "images" + slash
recording_directory = "recordings" + slash
LOG_MODE = 0  # 1 for time, 2 for frequency
MODEL_JSON = "model.json"
