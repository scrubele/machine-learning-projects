from random import random

from digits_sounds_recognizer import *
from file_adapters.files_adapter import *


def get_random_samples(file_names, samples_count_per_category):
    """
    get random samples for log testing.
    Args:
        file_names:
        samples_count_per_category:

    Returns:

    """
    checklist = np.zeros(samples_count_per_category * 10)
    final_list = []
    iteration_number = 0
    # Get a random sample for each category
    while 1:
        print("Iteration Number:", iteration_number)
        sample_names = random.sample(file_names, 10)
        for name in sample_names:
            category = int(name[0])
            if checklist[category] < samples_count_per_category:
                checklist[category] += 1
                final_list.append(name)
        if int(checklist.sum()) == (samples_count_per_category * 10):
            break
        iteration_number += 1
    print(final_list)
    return final_list


def generate_logs(in_dir, samples_count_per_category):
    """
    In Log mode capture one example of each class. Display it in time and frequency domain.
    Args:
        in_dir:
        samples_count_per_category:

    Returns:

    """
    file_names = sorted([f for f in os.listdir(in_dir) if '.wav' in f])
    random_files_list = get_random_samples(file_names, samples_count_per_category)
    current_logs_image_folder = os.path.join(log_image_folder, time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime()))
    if not os.path.exists(current_logs_image_folder):
        os.makedirs(current_logs_image_folder)
    for file_name in random_files_list:
        file_rate, file_data = wavfile.read(os.path.join(in_dir, file_name))
        if LOG_MODE == 1:  # Time Domain Signal
            plot_amplitude_graph(current_logs_image_folder, file_name, file_data)
        if LOG_MODE == 2:  # Frequency Domain Signals
            plot_frequency_graph(current_logs_image_folder, file_name, file_data, file_rate)
