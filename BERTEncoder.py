from transformers import BertModel
from torch import nn
import sys


class BERTEncoder(nn.Module):

    def __init__(self, model_name):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.output_size = 768

    def forward(self, captions):
        bert_outputs = self.bert(captions)
        # BERT gives us four outputs (see documentation). The first element
        # of this tuple is the output of the topmost Transformer block.
        top_layer_output = bert_outputs[0]
        # This tensor has the shape (n_docs, max_length, output_size)
        # where output_size is the size of the contextual representation
        # for each token (768 for a standard BERT).

        # Give a sign of life, because training is a bit time-consuming with this model...
        print('.', end='')
        sys.stdout.flush()

        return top_layer_output
