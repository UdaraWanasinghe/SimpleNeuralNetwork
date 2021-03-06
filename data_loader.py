import gzip
import numpy as np

TRAINING_IMAGES_PATH = 'data/train-images-idx3-ubyte.gz'
TRAINING_LABELS_PATH = 'data/train-labels-idx1-ubyte.gz'
TESTING_IMAGES_PATH = 'data/t10k-images-idx3-ubyte.gz'
TESTING_LABELS_PATH = 'data/t10k-labels-idx1-ubyte.gz'


def read_images(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of images
        image_count = int.from_bytes(f.read(4), 'big')
        # third 4 bytes is the row count
        row_count = int.from_bytes(f.read(4), 'big')
        # fourth 4 bytes is the column count
        column_count = int.from_bytes(f.read(4), 'big')
        # rest is the image pixel data, each pixel is stored as an unsigned byte
        # pixel values are 0 to 255
        image_data = f.read()
        images = np.frombuffer(image_data, dtype=np.uint8)\
            .reshape((image_count, row_count * column_count, 1))
        return images


def read_labels(path):
    with gzip.open(path, 'r') as f:
        # first 4 bytes is a magic number
        magic_number = int.from_bytes(f.read(4), 'big')
        # second 4 bytes is the number of labels
        label_count = int.from_bytes(f.read(4), 'big')
        # rest is the label data, each label is stored as unsigned byte
        # label values are 0 to 9
        label_data = f.read()
        labels = np.frombuffer(label_data, dtype=np.uint8)
        return labels


def prepare_training_data():
    images = read_images(TRAINING_IMAGES_PATH)
    labels = read_labels(TRAINING_LABELS_PATH)
    data = [(x, y) for x, y in zip(images, labels)]
    return data


def prepare_testing_data():
    images = read_images(TESTING_IMAGES_PATH)
    labels = read_labels(TESTING_LABELS_PATH)
    data = [(x, y) for x, y in zip(images, labels)]
    return data
