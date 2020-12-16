from collections import Counter
import os
import json

UNKNOWN = '<unk>'
START = '<start>'
END = '<end>'
PAD = '<pad>'


def load_vocab_from_json(data_name) -> dict:
    word_map_file = os.path.join('data/', 'VOCABMAP_' + data_name + '.json')
    with open(word_map_file, 'r') as j:
        return json.load(j)


class Vocabulary:
    """Manages the numerical encoding of the vocabulary."""

    def __init__(self, min_token_freq=5, caption_max_len=50):
        # String-to-integer mapping
        self.stoi = None

        # Minimal token frequency to be included in the vocabulary.
        self.min_token_freq = min_token_freq

        # Maximum allowed length of a caption
        self.caption_max_len = caption_max_len

    def build(self, captions):
        """Builds the vocabulary, based on a set of captions."""

        # Count token frequencies in captions
        token_freqs = Counter()
        for caption in captions:
            for token in caption:
                token_freqs.update(token)

        # Only keep tokens occurring more frequently than self.min_token_freq
        tokens = [token for token in token_freqs.keys() if token_freqs[token] > self.min_token_freq]

        # Build the token mapping. Add special tokens to the vocabulary:
        # <unk>, <start>, <end>, <pad>
        self.stoi = {k: v + 1 for v, k in enumerate(tokens)}
        self.stoi[UNKNOWN] = len(self.stoi) + 1
        self.stoi[START] = len(self.stoi) + 1
        self.stoi[END] = len(self.stoi) + 1
        self.stoi[PAD] = 0

    def encode(self, captions):
        """Encodes a set of captions. Save caption lengths for a mask."""
        enc = [[self.stoi[START]] +
               [self.stoi.get(token, self.stoi[UNKNOWN]) for token in caption] +
               [self.stoi[END]] +
               [self.stoi[PAD]] * (self.caption_max_len - len(caption)) for caption in captions]
        cap_len = [len(caption)+2 for caption in captions]
        return enc, cap_len

    def save_vocab_to_json(self, base_filename):
        with open(os.path.join('data/', 'VOCABMAP_' + base_filename + '.json'), 'w') as j:
            json.dump(self.stoi, j)

    def get_unknown_idx(self):
        """Returns the integer index of the special dummy word representing unknown words."""
        return self.stoi[UNKNOWN]

    def get_pad_idx(self):
        """Returns the integer index of the special padding dummy word."""
        return self.stoi[PAD]

    def get_start_idx(self):
        """Returns the integer index of the special start dummy word."""
        return self.stoi[START]

    def get_end_idx(self):
        """Returns the integer index of the special end dummy word."""
        return self.stoi[END]

    def __len__(self):
        return len(self.stoi)
