import torch
from torch import nn
import torchvision
from transformers import BertModel
from transformers import BertTokenizer
from utils import get_itos_from_stoi
import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Encoder(nn.Module):
    """
    Encoder.
    """

    def __init__(self, encoded_image_size=14):
        super(Encoder, self).__init__()
        self.enc_image_size = encoded_image_size

        resnet = torchvision.models.resnet101(pretrained=True)  # pretrained ImageNet ResNet-101

        # Remove linear and pool layers (since we're not doing classification)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)

        # Resize image to fixed size to allow input images of variable size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((encoded_image_size, encoded_image_size))

        self.fine_tune()

    def forward(self, images):
        """
        Forward propagation.

        :param images: images, a tensor of dimensions (batch_size, 3, image_size, image_size)
        :return: encoded images
        """
        out = self.resnet(images)  # (batch_size, 2048, image_size/32, image_size/32)
        out = self.adaptive_pool(out)  # (batch_size, 2048, encoded_image_size, encoded_image_size)
        out = out.permute(0, 2, 3, 1)  # (batch_size, encoded_image_size, encoded_image_size, 2048)
        return out

    def fine_tune(self, fine_tune=True):
        """
        Allow or prevent the computation of gradients for convolutional blocks 2 through 4 of the encoder.

        :param fine_tune: Allow?
        """
        for p in self.resnet.parameters():
            p.requires_grad = False
        # If fine-tuning, only fine-tune convolutional blocks 2 through 4
        for c in list(self.resnet.children())[5:]:
            for p in c.parameters():
                p.requires_grad = fine_tune


class Attention(nn.Module):
    """
    Attention Network.
    """

    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        """
        :param encoder_dim: feature size of encoded images
        :param decoder_dim: size of decoder's RNN
        :param attention_dim: size of the attention network
        """
        super(Attention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)  # linear layer to transform encoded image
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)  # linear layer to transform decoder's output
        self.full_att = nn.Linear(attention_dim, 1)  # linear layer to calculate values to be softmax-ed
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)  # softmax layer to calculate weights

    def forward(self, encoder_out, decoder_hidden):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :param decoder_hidden: previous decoder output, a tensor of dimension (batch_size, decoder_dim)
        :return: attention weighted encoding, weights
        """
        att1 = self.encoder_att(encoder_out)  # (batch_size, num_pixels, attention_dim)
        att2 = self.decoder_att(decoder_hidden)  # (batch_size, attention_dim)
        att = self.full_att(self.relu(att1 + att2.unsqueeze(1))).squeeze(2)  # (batch_size, num_pixels)
        alpha = self.softmax(att)  # (batch_size, num_pixels)
        attention_weighted_encoding = (encoder_out * alpha.unsqueeze(2)).sum(dim=1)  # (batch_size, encoder_dim)

        return attention_weighted_encoding, alpha


class DecoderWithAttention(nn.Module):
    """
    Decoder.
    """

    def __init__(self, attention_dim, embed_dim, decoder_dim, word_map, max_caption_length, encoder_dim=2048,
                 dropout=0.5, bert_model_name=None):
        """
        :param attention_dim: size of attention network
        :param embed_dim: embedding size
        :param decoder_dim: size of decoder's RNN
        :param vocab_size: size of vocabulary
        :param encoder_dim: feature size of encoded images
        :param dropout: dropout
        """
        super(DecoderWithAttention, self).__init__()
        self.fine_tune_embeddings_enabled = False
        self.vocab_size = len(word_map)
        self.max_caption_length = max_caption_length
        self.encoder_dim = encoder_dim
        self.attention_dim = attention_dim
        self.decoder_dim = decoder_dim
        self.dropout = dropout

        self.attention = Attention(encoder_dim, decoder_dim, attention_dim)  # attention network

        self.use_bert = False
        if bert_model_name:
            self.use_bert = True
            self.bert_tokenizer = BertTokenizer.from_pretrained(bert_model_name)
            self.BERT = BertModel.from_pretrained(bert_model_name)
            self.embed_dim = 768  # Default BERT embedding size
            self.itos = get_itos_from_stoi(word_map)
        else:
            self.embedding = nn.Embedding(self.vocab_size, embed_dim)
            self.embed_dim = embed_dim

        self.dropout = nn.Dropout(p=self.dropout)
        self.decode_step = nn.LSTMCell(self.embed_dim + encoder_dim, decoder_dim, bias=True)  # decoding LSTMCell
        self.init_h = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial hidden state of LSTMCell
        self.init_c = nn.Linear(encoder_dim, decoder_dim)  # linear layer to find initial cell state of LSTMCell
        self.f_beta = nn.Linear(decoder_dim, encoder_dim)  # linear layer to create a sigmoid-activated gate
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(decoder_dim, self.vocab_size)  # linear layer to find scores over vocabulary
        self.init_weights(bert_model_name)  # initialize some layers with the uniform distribution

    def init_weights(self, bert_model_name=None):
        """
        Initializes some parameters with values from the uniform distribution, for easier convergence.
        """
        if not bert_model_name:
            self.embedding.weight.data.uniform_(-0.1, 0.1)
        self.fc.bias.data.fill_(0)
        self.fc.weight.data.uniform_(-0.1, 0.1)

    def load_pretrained_embeddings(self, embeddings):
        """
        Loads embedding layer with pre-trained embeddings.

        :param embeddings: pre-trained embeddings
        """
        self.embedding.weight = nn.Parameter(embeddings)

    def fine_tune_embeddings(self, fine_tune):
        """
        Allow fine-tuning of embedding layer? (Only makes sense to not-allow if using pre-trained embeddings).

        :param fine_tune: Allow?
        """
        self.fine_tune_embeddings_enabled = fine_tune
        if not self.use_bert:
            for p in self.embedding.parameters():
                p.requires_grad = fine_tune

    def init_hidden_state(self, encoder_out):
        """
        Creates the initial hidden and cell states for the decoder's LSTM based on the encoded images.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, num_pixels, encoder_dim)
        :return: hidden state, cell state
        """
        mean_encoder_out = encoder_out.mean(dim=1)
        h = self.init_h(mean_encoder_out)  # (batch_size, decoder_dim)
        c = self.init_c(mean_encoder_out)
        return h, c

    def forward(self, encoder_out, encoded_captions, caption_lengths):
        """
        Forward propagation.

        :param encoder_out: encoded images, a tensor of dimension (batch_size, enc_image_size, enc_image_size, encoder_dim)
        :param encoded_captions: encoded captions, a tensor of dimension (batch_size, max_caption_length)
        :param caption_lengths: caption lengths, a tensor of dimension (batch_size, 1)
        :return: scores for vocabulary, sorted encoded captions, decode lengths, weights, sort indices
        """

        batch_size = encoder_out.size(0)
        encoder_dim = encoder_out.size(-1)
        vocab_size = self.vocab_size

        # Flatten image
        encoder_out = encoder_out.view(batch_size, -1, encoder_dim)  # (batch_size, num_pixels, encoder_dim)
        num_pixels = encoder_out.size(1)

        # Sort input data by decreasing lengths; why? apparent below
        caption_lengths, sort_ind = caption_lengths.squeeze(1).sort(dim=0, descending=True)
        encoder_out = encoder_out[sort_ind]
        encoded_captions = encoded_captions[sort_ind]

        # Embedding
        if self.use_bert:
            embeddings = self.load_bert_embeddings(encoded_captions)
            # Give a sign of life, because training is a bit time-consuming with this model...
            print('.', end='')
            sys.stdout.flush()
        else:
            embeddings = self.embedding(encoded_captions)  # (batch_size, max_caption_length, embed_dim)

        # Initialize LSTM state
        h, c = self.init_hidden_state(encoder_out)  # (batch_size, decoder_dim)

        # We won't decode at the <end> position, since we've finished generating as soon as we generate <end>
        # So, decoding lengths are actual lengths - 1
        decode_lengths = (caption_lengths - 1).tolist()

        # Create tensors to hold word predicion scores and alphas
        predictions = torch.zeros(batch_size, max(decode_lengths), vocab_size).to(device)
        alphas = torch.zeros(batch_size, max(decode_lengths), num_pixels).to(device)

        # At each time-step, decode by
        # attention-weighing the encoder's output based on the decoder's previous hidden state output
        # then generate a new word in the decoder with the previous word and the attention weighted encoding
        for t in range(max(decode_lengths)):
            batch_size_t = sum([l > t for l in decode_lengths])
            attention_weighted_encoding, alpha = self.attention(encoder_out[:batch_size_t],
                                                                h[:batch_size_t])
            gate = self.sigmoid(self.f_beta(h[:batch_size_t]))  # gating scalar, (batch_size_t, encoder_dim)
            attention_weighted_encoding = gate * attention_weighted_encoding
            h, c = self.decode_step(
                torch.cat([embeddings[:batch_size_t, t, :], attention_weighted_encoding], dim=1),
                (h[:batch_size_t], c[:batch_size_t]))  # (batch_size_t, decoder_dim)
            preds = self.fc(self.dropout(h))  # (batch_size_t, vocab_size)
            predictions[:batch_size_t, t, :] = preds
            alphas[:batch_size_t, t, :] = alpha

        return predictions, encoded_captions, decode_lengths, alphas, sort_ind

    def load_bert_embeddings(self, encoded_captions):
        embeddings = []
        for enc_caption in encoded_captions:
            # Convert encoded caption to text
            caption = self.get_caption_from_enc_caption(enc_caption)

            # Tokenize and encode text
            tokenized_caption = self.bert_tokenizer.tokenize(caption)
            indexed_tokens = self.bert_tokenizer.convert_tokens_to_ids(tokenized_caption)
            tokens_tensor = torch.tensor([indexed_tokens]).to(device)

            # Get word embeddings
            if self.fine_tune_embeddings_enabled:
                bert_output = self.BERT(tokens_tensor)
            else:
                with torch.no_grad():
                    bert_output = self.BERT(tokens_tensor)
            bert_embedding = bert_output[0].squeeze(0)

            split_caption = caption.split()
            tokens_embedding = []
            j = 0

            for full_token in split_caption:
                curr_token = ''
                x = 0
                for i, _ in enumerate(tokenized_caption):
                    token = tokenized_caption[i + j]
                    piece_embedding = bert_embedding[i + j]

                    # full token
                    if token == full_token and curr_token == '':
                        tokens_embedding.append(piece_embedding)
                        j += 1
                        break
                    else:  # partial token
                        x += 1

                        if curr_token == '':
                            tokens_embedding.append(piece_embedding)
                            curr_token += token.replace('#', '')
                        else:
                            tokens_embedding[-1] = torch.add(tokens_embedding[-1], piece_embedding)
                            curr_token += token.replace('#', '')

                            if curr_token == full_token:  # end of partial
                                j += x
                                break

            cap_embedding = torch.stack(tokens_embedding)
            embeddings.append(cap_embedding)

        embeddings = torch.stack(embeddings)

        return embeddings

    def get_caption_from_enc_caption(self, enc_caption):
        caption = []
        for enc_word in enc_caption:
            word = self.itos[enc_word.item()]
            if word == '<start>':
                word = self.bert_tokenizer.cls_token
            if word == '<end>':
                word = self.bert_tokenizer.sep_token
            if word == '<pad>':
                word = self.bert_tokenizer.pad_token
            if word == '<unk>':
                word = self.bert_tokenizer.unk_token
            caption.append(word)
        return " ".join(caption)
