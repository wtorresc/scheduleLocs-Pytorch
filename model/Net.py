import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

import numpy as np
import pdb

class LocationRNN(nn.Module):
    """
    The RNN model that will be used to perform Sentiment analysis.
    """

    def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers,
                 rnn_model='GRU', drop_prob=0.5, use_last=True, embedding_tensor=None,
                 padding_index=0, batch_first=True):
        """
        Initialize the model by setting up the layers.

        Args:
            vocab_size: vocab size
            embedding_dim: embedding size
            num_output: number of output (classes)
            rnn_model:  LSTM or GRU
            use_last:  bool
            embedding_tensor:
            padding_index:
            hidden_dim: hidden size of rnn module
            n_layers:  number of layers in rnn module
            batch_first: batch first option

        """
        super(LocationRNN, self).__init__()

        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_dim = hidden_dim
        self.use_last = use_last
        self.rnn_type = rnn_model
        num_classes = 5
        self.num_classes = 5

        self.encoder = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_index)
        
        # dropout layer
        self.drop_en = nn.Dropout(p=0.5)
        
        # embedding and LSTM layers
        if rnn_model == 'LSTM':
            self.rnn = nn.LSTM( input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob,
                                batch_first=True, bidirectional=True)
        elif rnn_model == 'GRU':
            self.rnn = nn.GRU( input_size=embedding_dim, hidden_size=hidden_dim, num_layers=n_layers, dropout=drop_prob,
                                batch_first=True, bidirectional=True)
        else:
            raise LookupError(' only support LSTM and GRU')
        
        # dropout layer
        self.drop_de = nn.Dropout(p=0.5)

        self.decoder = nn.Linear(2*hidden_dim, num_classes)

    def init_weights(self):
        initrange = 0.1

        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, x, seq_lengths, hidden=None):
        '''
        Args:
            x: (batch, time_step, input_size)
        Returns:
            num_output size
        '''

        # if not seq_lengths.is_cuda:
        #     seq_lengths = torch.from_numpy(seq_lengths).to(device)

        # embeddings
        x_embed = self.encoder(x)
        x_embed = self.drop_en(x_embed)
        packed_input = pack_padded_sequence(x_embed, seq_lengths, batch_first=True)

        # r_out shape (batch, time_step, output_size)
        # None is for initial hidden state
        packed_output, hidden = self.rnn(packed_input, hidden)
        out_rnn, _ = pad_packed_sequence(packed_output, batch_first=True)
        # pdb.set_trace()
        out_rnn = self.drop_de(out_rnn)
        decoded = self.decoder(out_rnn.contiguous().view(out_rnn.size(0)*out_rnn.size(1), out_rnn.size(2)))    
        # pdb.set_trace()
        return decoded.view(out_rnn.size(0), out_rnn.size(1), decoded.size(1)), hidden
    
    def init_hidden(self, batch_size):
        ''' Initializes hidden state '''
        # Create two new tensors with sizes n_layers x batch_size x hidden_dim,
        # initialized to zero, for hidden state and cell state of LSTM
        weight = next(self.parameters())

        if self.rnn.bidirectional:
            directions = 2
        else:
            directions = 1

        
        if self.rnn_type == 'LSTM':
            hidden = (weight.new_zeros(self.n_layers * directions, batch_size, self.hidden_dim),
                      weight.new_zeros(self.n_layers * directions, batch_size, self.hidden_dim))
        else:
            hidden = weight.new_zeros(self.n_layers * directions, batch_size, self.hidden_dim)
        
        return hidden