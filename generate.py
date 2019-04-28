"""Generate the output"""

import argparse
import io
import os
import pdb
import json
from collections import Counter, OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, PackedSequence

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence

from model.Net import LocationRNN
import model.data_loader as data_loader
import utils


def get_args():

    parser = argparse.ArgumentParser(description='Uniformat Classification')

    parser.add_argument('--test-batchsize',
                        '-TB',
                        type=int,
                        default=256,
                        metavar='N',
                        help='Define test mini-batch size (default: 500)')

    parser.add_argument('--no-cuda',
                        action='store_true',
                        default=False,
                        help='disables CUDA training')

    parser.add_argument('--log-interval',
                        type=int,
                        default=100,
                        metavar='N',
                        help='disables CUDA training')

    parser.add_argument('--num-workers',
                       '-NW',
                        default=4,
                        type=int,
                        metavar='N',
                        help='number of data loading workers (default: 4)')

    parser.add_argument('--data-dir',
                        default='./data',
                        help="Dataset folder (default : ./data)")

    parser.add_argument('--classses-file',
                        default='./data/uniformat.json',
                        help="Dataset folder (default : ./data/uniformat.json)")

    parser.add_argument('--vocab-size',
                        type=int,
                        default=2000,
                        metavar='N',
                        help='Size of vocabulary (default: 2000)')

    parser.add_argument('--output-size',
                        type=int,
                        default=1,
                        metavar='N',
                        help='Size of desired output (default: 1)')
    
    parser.add_argument('--saved-model',
                        type=str,
                        default='./locations_model.pth.tar',
                        help='Saved model')

    parser.add_argument('--embedding-dim',
                        type=int,
                        default=150,
                        metavar='N',
                        help='Size of embeddings (default: 150)')

    parser.add_argument('--hidden-dim',
                        type=int,
                        default=256,
                        metavar='N',
                        help='Number of units in the hidden layers of LSTM cells (default: 256)')

    parser.add_argument('--n-layers',
                        type=int,
                        default=2,
                        metavar='N',
                        help='Number of LSTM layers in the network (default: 2)') 

    args = parser.parse_args()

    return args



def predict(args, model, device, test_dataloader, outfile=None):
    # set model to test mode
    model.eval()

    # summary for current test loop and a running average object for loss
    # hidden = model.init_hidden(args.test_batchsize)
    hidden = None

    # criterion.reduction = 'sum'
    # prediction_arr = np.array((6,len(test_dataloader.dataset)))

    with torch.no_grad():
        for batch_idx, (x_test_batch, seq_lengths) in enumerate(test_dataloader):
              
            x_test_batch, seq_lengths, unperm_idx = sort_batch(x_test_batch, seq_lengths)
            # max_batch_size = int(seq_lengths[0])
            data = x_test_batch.to(device)

            # if len(test_dataloader) - 1 == batch_idx:
            #     hidden = None

            output, hidden = model(data, seq_lengths, hidden)

            # pack_padded_sequence is a good trick to do this
            output, batch_sizes = pack_padded_sequence(output, seq_lengths, batch_first=True)

            prediction = output.max(1)[1]
            
            # print(prediction)
            prediction_matrix, _ = pad_packed_sequence(PackedSequence(prediction,batch_sizes))
            prediction_matrix = prediction_matrix[:,unperm_idx].cpu().numpy()
            # prediction_matrix = prediction.view(max_batch_size,-1).cpu().numpy()
            prediction_arr = [prediction_matrix[:seq_length_i, idx].tolist() for idx, seq_length_i in enumerate(seq_lengths[unperm_idx])]
            # print(prediction_arr)
            # pdb.set_trace()
            # hidden = repackage_hidden(hidden)

    if outfile:
        np.savetxt(outfile, prediction_arr)


def sort_batch(samples, lengths):
    
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = samples[perm_idx]
    _, unperm_idx = perm_idx.sort(0)
    # targ_tensor = labels[perm_idx]

    return seq_tensor, seq_lengths.cpu().numpy(), unperm_idx


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)


if __name__ == '__main__':
    """ Main function """
    args = get_args()

    vocab = np.load('./preprocessed_data/id2word_dictionary.npy') # ordered vocab
    word_to_id = {token:idx for idx, token in enumerate(vocab, 1)}
    word_to_id["<UNK>"] = 0

    vocab_size = 2000

    path='./activityList.json'
    x_predict = []

    with io.open(os.path.expanduser(path), encoding="utf8") as f:

        reader = json.load(f)
        reader = reader['content']

        for line in reader:
            x_predict.append(utils._spacy_tokenize(line['activityName']))

    x_predict_token_ids = [[word_to_id.get(token,0) for token in x] for x in x_predict]
    x_token_ids = x_predict_token_ids
    x_token_ids = list(map(torch.LongTensor, x_token_ids))
    seq_lengths = torch.LongTensor(list(map(len, x_token_ids)))

    padded_tensor = torch.nn.utils.rnn.pad_sequence(x_token_ids, batch_first=True) 

    # Set to zero indices above the vocab size
    padded_tensor[padded_tensor > vocab_size] = 0       

    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # # Run on GPU or CPU
    device = torch.device("cuda" if use_cuda else "cpu")

    # # Fetch data
    # dataloaders = data_loader.fetch_data(['predict'], args)
    # predict_dataloader = dataloaders['predict']
    predict_dataloader = [(padded_tensor,seq_lengths)]

    args.vocab_size += 1 

    # idx_to_class, _ =  data_loader.get_classes(args.classses_file)

    # Load Model
    model = LocationRNN(args.vocab_size, args.output_size, args.embedding_dim, args.hidden_dim, args.n_layers)
    model.load_state_dict(torch.load(args.saved_model))
    model = model.to(device)

    predict(args, model, device, predict_dataloader)
