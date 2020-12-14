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
    word_freq = Counter()

    with open(dataset_split_path, 'r') as f:
        dataset = json.load(f)

    for image in dataset["images"]:
        captions = process_captions(caption_max_len, image, word_freq)

        if len(captions) == 0:
            continue

        path = os.path.join(dataset_image_path, image['filename'])

        


def process_captions(caption_max_len, image, word_freq):
    captions = []
    for caption in image['sentences']:
        tokens = caption["tokens"]
        word_freq.update(tokens)
        if len(tokens) <= caption_max_len:
            captions.append(tokens)
    return captions


