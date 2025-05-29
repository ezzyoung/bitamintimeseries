import os
import numpy as np
from pyedflib import highlevel
import matplotlib.pyplot as plt

# Helper functions for searching files
def search_annotations_edf(dirname):
    filenames = [file for file in os.listdir(dirname) if file.endswith("Hypnogram.edf")]
    return filenames

def search_signals_npy(dirname):
    filenames = [file for file in os.listdir(dirname) if file.endswith(".npy")]
    return filenames

def search_correct_annotations_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    matched_files = [file for file in file_list if search_filename in file and file.endswith(".npy")]
    if not matched_files:
        raise FileNotFoundError(f"No matching annotation file found for {filename}")
    return matched_files[0]

def search_correct_signals_npy(dirname, filename):
    search_filename = filename.split('-')[0][:-2]
    file_list = os.listdir(dirname)
    matched_files = [file for file in file_list if search_filename in file and file.endswith(".npy")]
    if not matched_files:
        raise FileNotFoundError(f"No matching signal file found for {filename}")
    return matched_files[0]

# Main preprocessing logic
path = "./sleep-cassette/"
annotations_edf_list = search_annotations_edf(path)
save_annotations_path = os.path.join(path, 'annotations/')
os.makedirs(save_annotations_path, exist_ok=True)


epoch_size = 30
sample_rate = 100

for filename in annotations_edf_list:
    try:
        annotations_filepath = os.path.join(path, filename)
        _, _, annotations_header = highlevel.read_edf(annotations_filepath)

        label = []
        for ann in annotations_header['annotations']:
            start = ann[0]
            length = int(np.ceil(float(ann[1]) / epoch_size))  # 30초 단위로 나눔
            stage = ann[2]

            if stage == 'Sleep stage W':
                label.extend([0] * length)
            elif stage == 'Sleep stage 1':
                label.extend([1] * length)
            elif stage == 'Sleep stage 2':
                label.extend([2] * length)
            elif stage == 'Sleep stage 3' or stage == 'Sleep stage 4':
                label.extend([3] * length)
            elif stage == 'Sleep stage R':
                label.extend([4] * length)
            else:
                label.extend([5] * length)

        # 변환된 데이터를 저장
        file_name = os.path.splitext(filename)[0]
        np.save(os.path.join(save_annotations_path, file_name), label)

        print(f"Converted {filename} to {file_name}.npy")
    except Exception as e:
        print(f"Error processing file {filename}: {e}")
