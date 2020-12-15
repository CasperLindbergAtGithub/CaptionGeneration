import json
from Vocabulary import Vocabulary
import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from random import choice, sample

dataset_image_path = 'data/Flicker8k_Dataset'
dataset_split_path = 'data/dataset_flickr8k.json'
data_folder = 'data/'


def preprocess_data(min_token_freq=5, caption_max_len=50, captions_per_image=5):
    """
    :param captions_per_image: the number of captions for each image
    :param min_token_freq: words occurring less frequently than this threshold are binned as <unk>
    :param caption_max_len: maximum allowed length of captions
    """

    image_train = []
    image_validation = []
    image_test = []
    captions_train = []
    captions_validation = []
    captions_test = []

    # Read data
    dataset = load_dataset()

    # Process each image in the dataset
    for image in dataset["images"]:

        # Read and process the captions for the image
        processed_captions = process_captions(caption_max_len, image)

        if len(processed_captions) == 0:
            continue

        path = os.path.join(dataset_image_path, image['filename'])

        train_val_test_split(processed_captions, captions_test, captions_train, captions_validation, image, image_test,
                             image_train, image_validation, path)

    assert_valid_train_val_test_split(captions_test, captions_train, captions_validation, image_test, image_train,
                                      image_validation)

    all_captions = captions_train + captions_validation + captions_test
    vocab = create_vocabulary(all_captions, min_token_freq, caption_max_len)

    base_filename = save_vocab_map_to_json(captions_per_image, min_token_freq, vocab.stoi)

    for image_paths, image_captions, split in [(image_train, captions_train, 'TRAIN'),
                                               (image_validation, captions_validation, 'VAL'),
                                               (image_test, captions_test, 'TEST')]:
        with h5py.File(os.path.join(data_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Note the number of captions we are sampling per image
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 to store images
            images = h.require_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')

            print("\nReading %s images and captions, storing to file...\n" % split)

            cap_len, enc_cap = encode_captions_save_images(captions_per_image, image_captions,
                                                           image_paths, images, vocab)

            # Sanity check
            assert images.shape[0] * captions_per_image == len(enc_cap) == len(cap_len)

            save_encoded_captions_and_lengths_to_json(base_filename, cap_len, enc_cap, split)


def load_dataset():
    with open(dataset_split_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def save_encoded_captions_and_lengths_to_json(base_filename, cap_len, enc_cap, split):
    with open(os.path.join(data_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
        json.dump(enc_cap, j)
    with open(os.path.join(data_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
        json.dump(cap_len, j)


def encode_captions_save_images(captions_per_image, image_captions, image_paths, images, vocab):
    encoded_captions = []
    caption_lengths = []
    for i, path in enumerate(tqdm(image_paths)):
        captions = sample_captions(i, captions_per_image, image_captions)
        img = read_format_image(i, image_paths)

        # Save image to HDF5 file
        images[i] = img

        # Encode captions
        [enc, cap_len] = vocab.encode(captions)
        encoded_captions.extend(enc)
        caption_lengths.extend(cap_len)
    return caption_lengths, encoded_captions


def read_format_image(i, image_paths):
    img = cv2.imread(image_paths[i])
    if len(img.shape) == 2:  # Grey-scale image
        img = img[:, :, np.newaxis]
        img = np.concatenate([img, img, img], axis=2)
    img = cv2.resize(img, (256, 256))
    img = img.transpose(2, 0, 1)
    assert img.shape == (3, 256, 256)
    assert np.max(img) <= 255

    return img


def sample_captions(i, captions_per_image, image_captions) -> list:
    if len(image_captions[i]) < captions_per_image:  # with replacement
        captions = image_captions[i] + [choice(image_captions[i])
                                        for _ in range(captions_per_image - len(image_captions[i]))]
    else:  # without replacement
        captions = sample(image_captions[i], k=captions_per_image)
    assert len(captions) == captions_per_image  # Sanity check

    return captions


def process_captions(caption_max_len, image):
    captions = []
    for caption in image['sentences']:
        tokens = caption["tokens"]
        if len(tokens) <= caption_max_len:
            captions.append(tokens)
    return captions


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


def create_vocabulary(captions, min_token_freq, caption_max_len) -> Vocabulary:
    vocab = Vocabulary(min_token_freq, caption_max_len)
    vocab.build(captions)
    return vocab


def save_vocab_map_to_json(captions_per_image, min_token_freq, vocab_map) -> str:
    base_filename = 'flickr8k' + '_' + str(captions_per_image) + '_cap_per_img_' + str(
        min_token_freq) + '_min_token_freq'

    with open(os.path.join('data/', 'VOCABMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(vocab_map, j)

    return base_filename
