import json
from Vocabulary import Vocabulary
import os
import h5py
import cv2
import numpy as np
from tqdm import tqdm
from random import choice, sample
import torch

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

    base_filename = 'flickr8k' + '_' + str(captions_per_image) + '_cap_per_img_' + str(
        min_token_freq) + '_min_token_freq'

    vocab.save_vocab_to_json(base_filename)

    # Save images to HDF5 file, and captions and their lengths to JSON files.
    for image_paths, image_captions, split in [(image_train, captions_train, 'TRAIN'),
                                               (image_validation, captions_validation, 'VAL'),
                                               (image_test, captions_test, 'TEST')]:

        print("\nReading %s images and captions, storing to file...\n" % split)

        with h5py.File(os.path.join(data_folder, split + '_IMAGES_' + base_filename + '.hdf5'), 'a') as h:
            # Note the number of captions we are sampling per image - see sample_captions()
            h.attrs['captions_per_image'] = captions_per_image

            # Create dataset inside HDF5 to store images.
            images = h.require_dataset('images', (len(image_paths), 3, 256, 256), dtype='uint8')

            for i, path in enumerate(tqdm(image_paths)):
                captions = sample_captions(i, captions_per_image, image_captions)
                img = read_format_image(i, image_paths)

                # Save image to HDF5 file
                images[i] = img

                # Encode captions and get their lengths
                (encoded_captions, caption_lengths) = vocab.encode(captions)

                # Save to JSON
                save_encoded_captions_and_lengths_to_json(base_filename, split, encoded_captions,
                                                          caption_lengths)


def load_dataset():
    with open(dataset_split_path, 'r') as f:
        dataset = json.load(f)
    return dataset


def read_format_image(i, image_paths):
    """
    ImageNet requires 3-channel 256x256 images, so images must be resized.
    """
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
    """
    If number of captions for an image is less than captions_per_image, we need to sample from
    the existing captions to fill out.
    """
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


def save_vocab_map_to_json(base_filename, vocab_map):
    with open(os.path.join(data_folder, 'VOCABMAP_' + base_filename + '.json'), 'w') as j:
        json.dump(vocab_map, j)


def save_encoded_captions_and_lengths_to_json(base_filename, split, encoded_captions, caption_lengths):
    with open(os.path.join(data_folder, split + '_CAPTIONS_' + base_filename + '.json'), 'w') as j:
        json.dump(encoded_captions, j)

    with open(os.path.join(data_folder, split + '_CAPLENS_' + base_filename + '.json'), 'w') as j:
        json.dump(caption_lengths, j)


def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrinks learning rate by a specified factor.
    :param optimizer: optimizer whose learning rate must be shrunk.
    :param shrink_factor: factor in interval (0, 1) to multiply learning rate with.
    """

    print("\nDECAYING learning rate.")
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * shrink_factor
    print("The new learning rate is %f\n" % (optimizer.param_groups[0]['lr'],))


def save_checkpoint(data_name, epoch, epochs_since_improvement, encoder, decoder, encoder_optimizer, decoder_optimizer,
                    bleu4, is_best):
    """
    Saves model checkpoint.
    :param data_name: base name of processed dataset
    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement in BLEU-4 score
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update encoder's weights, if fine-tuning
    :param decoder_optimizer: optimizer to update decoder's weights
    :param bleu4: validation BLEU-4 score for this epoch
    :param is_best: is this checkpoint the best so far?
    """
    state = {'epoch': epoch,
             'epochs_since_improvement': epochs_since_improvement,
             'bleu-4': bleu4,
             'encoder': encoder,
             'decoder': decoder,
             'encoder_optimizer': encoder_optimizer,
             'decoder_optimizer': decoder_optimizer}
    filename = 'checkpoint_' + data_name + '.pth.tar'
    torch.save(state, filename)
    # If this checkpoint is the best so far, store a copy so it doesn't get overwritten by a worse checkpoint
    if is_best:
        torch.save(state, 'BEST_' + filename)


class AverageMeter(object):
    """
    Keeps track of most recent, average, sum, and count of a metric.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def clip_gradient(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.
    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def accuracy(scores, targets, k):
    """
    Computes top-k accuracy, from predicted and true labels.
    :param scores: scores from the model
    :param targets: true labels
    :param k: k in top-k accuracy
    :return: top-k accuracy
    """

    batch_size = targets.size(0)
    _, ind = scores.topk(k, 1, True, True)
    correct = ind.eq(targets.view(-1, 1).expand_as(ind))
    correct_total = correct.view(-1).float().sum()  # 0D tensor
    return correct_total.item() * (100.0 / batch_size)