import json
from collections import Counter
import os

dataset_image_path = 'data/Flicker8k_Dataset'
dataset_split_path = 'data/dataset_flickr8k.json'


def preprocess_data(caption_max_len: int = 50, min_word_freq: int = 5):
    """
    :param caption_max_len: maximum allowed length of captions
    :param min_word_freq: words occuring less frequently than this threshold are binned as <unk>
    """

    # Store images and captions
    image_train = []
    image_validation = []
    image_test = []
    captions_train = []
    captions_validation = []
    captions_test = []

    with open(dataset_split_path, 'r') as f:
        dataset = json.load(f)

    for image in dataset["images"]:
        captions = process_captions(caption_max_len, image)

        if len(captions) == 0:
            continue

        path = os.path.join(dataset_image_path, image['filename'])

        train_val_test_split(captions, captions_test, captions_train, captions_validation, image, image_test,
                             image_train, image_validation, path)

        assert_valid_train_val_test_split(captions_test, captions_train, captions_validation, image_test, image_train,
                                          image_validation)


def train_val_test_split(captions, captions_test, captions_train, captions_validation, image, image_test, image_train,
                         image_validation, path):
    if image['split'] in {'train'}:
        image_train.append(path)
        captions_train.append(captions)
    elif image['split'] in {'val'}:
        image_validation.append(path)
        captions_validation.append(captions)
    elif image['split'] in {'test'}:
        image_test.append(path)
        captions_test.append(captions)


def assert_valid_train_val_test_split(captions_test, captions_train, captions_validation, image_test, image_train,
                                      image_validation):
    assert len(image_train) == len(captions_train)
    assert len(image_validation) == len(captions_validation)
    assert len(image_test) == len(captions_test)


def process_captions(caption_max_len, image):
    captions = []
    for caption in image['sentences']:
        tokens = caption["tokens"]
        if len(tokens) <= caption_max_len:
            captions.append(tokens)
    return captions
