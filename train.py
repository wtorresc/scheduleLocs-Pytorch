"""Train the model"""

import argparse
import logging
import time
import math
import os
import pdb

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
from tqdm import tqdm

from model.Net_language_model import LocationRNN
import model.data_loader as data_loader

from tensorboardX import SummaryWriter
import logging
# from evaluate import evaluate

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(message)s',
                    datefmt='%m/%d/%Y %I:%M:%S %p')

# if not os.path.exists('/run_logs/run1'):
#     os.makedirs('/run_logs/run1')
# else 

writer = SummaryWriter('run_logs/run7')

def get_args():

    parser = argparse.ArgumentParser(description='Uniformat Classification')

    parser.add_argument('--batchsize',
                        '-B',
                        type=int,
                        default=8,
                        metavar='N',
                        help='Define mini-batch size (default: 8)')

    parser.add_argument('--test-batchsize',
                        '-TB',
                        type=int,
                        default=256,
                        metavar='N',
                        help='Define test mini-batch size (default: 500)')

    parser.add_argument('--epochs',
                        '-E',
                        type=int,
                        default=25,
                        help='Define the number of epochs (default: 10)')

    parser.add_argument('--learning-rate',
                        '-LR',
                        type=float,
                        default=0.001,
                        help='Define the learning rate (default: 0.001)')

    parser.add_argument('--weight-decay',
                        '-WD',
                        type=float,
                        default=0.000005,
                        help='Define the weight decay (default: 0.0005)')

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

    parser.add_argument('--embedding-dim',
                        type=int,
                        default=150,
                        metavar='N',
                        help='Size of embeddings (default: 350)')

    parser.add_argument('--hidden-dim',
                        type=int,
                        default=256,
                        metavar='N',
                        help='Number of units in the hidden layers of LSTM cells (default: 256)')

    parser.add_argument('--n-layers',
                        type=int,
                        default=3,
                        metavar='N',
                        help='Number of LSTM layers in the network (default: 3)')
    
    parser.add_argument('--resume',
                        '-R',
                        default='./checkpoint.pth.tar',
                        type=str,
                        help='path to latest checkpoint (default: none)')

    parser.add_argument('--clip',
                        '-cl',
                        default=0.25,
                        type=float,
                        metavar='PATH',
                        help='gradient clipping(default: 0.25)')  

    args = parser.parse_args()

    return args


def train(args, model, device, dataloader, optimizer, criterion, epoch, idx_to_class):
    # set model to training mode
    model.train()

    # summary for current training loop and a running average object for loss
    train_accu = []
    train_loss = []
    
    # initialize hidden state
    hidden = model.init_hidden(args.batchsize)    
    # pdb.set_trace()

    # Use tqdm for progress bar
    with tqdm(total=len(dataloader.dataset), unit="batch") as t:
        for batch_idx, (x_train_batch, y_train_batch, seq_lengths) in enumerate(dataloader):

            x_train_batch, _ , seq_lengths = sort_batch(x_train_batch, y_train_batch, seq_lengths)
            max_batch_size = int(seq_lengths[0])

            data, target = x_train_batch.to(device), x_train_batch[:,:max_batch_size].to(device)
            # pdb.set_trace()
            

            # Starting each batch, we detach the hidden state from how it was previously produced.
            # If we didn't, the model would try backpropagating all the way to start of the dataset.
            hidden = repackage_hidden(hidden)

            # PyTorch "accumulates gradients", so we need to set the stored
            # gradients to zero when there’s a new batch of data.
            optimizer.zero_grad()
            # pdb.set_trace()

            #Forward propagation of the model, i.e. calculate the hidden
            # units and the output.
            output, hidden  = model(data, seq_lengths, hidden)
            
            #The objective function is the negative log-likelihood function.
            # pdb.set_trace()
            loss = criterion(output.view(-1, args.vocab_size)[:-1,:], target.view(-1)[1:])

            train_loss.append(loss.item())
            # pdb.set_trace()

            niter = epoch*len(dataloader) + batch_idx
            writer.add_scalar('loss/loss', loss.item(), niter)

            #This calculates the gradients (via backpropagation)
            loss.backward()

            prediction = output.view(-1, args.vocab_size)[:-1,:].max(1)[1]

            # pdb.set_trace()
            acc = prediction.eq(target.view(-1)[1:])

            accuracy = float(acc.sum()) / float(acc.numel())#float(args.batchsize)
            train_accu.append(accuracy)

            writer.add_scalar('acc/acc', accuracy, niter)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)

            # performs updates using calculated gradients
            optimizer.step()

            # manual version of optimizer.step -> p.data = p.data + (-lr * p.grad.data)
            # for p in model.parameters():
            #     p.data.add_(-lr, p.grad.data)
           
           # t.set_description('Epoch: {}/{}'.format(epoch, args.epochs))
            t.set_postfix(loss='{:05.3f}'.format(loss.item()),
                          acc='{:04.2f}%'.format(np.mean(train_accu) * 100))
            t.update(args.batchsize)

def test(args, model, device, test_dataloader, criterion, idx_to_class, outfile=None):
    # set model to test mode
    model.eval()

    # summary for current test loop and a running average object for loss
    test_loss = 0
    correct = 0
    total_elements = 0
    hidden = model.init_hidden(args.test_batchsize)

    criterion.reduction = 'sum'
    prediction_arr = np.array((6,len(test_dataloader.dataset)))

    with torch.no_grad():
        for batch_idx, (x_test_batch, y_test_batch, seq_lengths) in enumerate(test_dataloader):
            
            x_test_batch, y_test_batch, seq_lengths = sort_batch(x_test_batch, y_test_batch, seq_lengths)
            max_batch_size = int(seq_lengths[0])
            data, target = x_test_batch.to(device), x_test_batch[:,:max_batch_size].to(device)

            if len(test_dataloader) - 1 == batch_idx:
                hidden = None

            output, hidden = model(data, seq_lengths, hidden)

            loss = criterion(output.view(-1, args.vocab_size)[:-1,:], target.view(-1)[1:])

            test_loss += loss.item()

            niter = batch_idx
            writer.add_scalar('loss', loss.item(), niter)

            prediction = output.view(-1, args.vocab_size)[:-1,:].max(1)[1]

            acc = prediction.eq(target.view(-1)[1:])

            accuracy = acc.sum() / float(acc.numel()) #float(args.test_batchsize))

            correct += acc.sum()
            total_elements += acc.numel()

            writer.add_scalar('acc', accuracy, niter)

            hidden = repackage_hidden(hidden)

    if outfile:
        np.savetxt(outfile, prediction_arr)

    test_loss /= total_elements #len(test_dataloader.dataset) #(args.test_batchsize*len(test_dataloader))

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, total_elements, #len(test_dataloader.dataset), #args.test_batchsize*len(test_dataloader),
        100. * correct / total_elements)) # len(test_dataloader.dataset))) #(args.test_batchsize*len(test_dataloader))))


def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def sort_batch(samples, labels, lengths):
    
    seq_lengths, perm_idx = lengths.sort(0, descending=True)
    seq_tensor = samples[perm_idx]
    targ_tensor = labels[perm_idx]

    return seq_tensor, targ_tensor, seq_lengths.cpu().numpy()

def plot_confusion(conf_matrix, idx_to_class, outfile = None):
    # Set up plot
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(conf_matrix, vmin=0, vmax=1)
    v = np.linspace(0, 1, 5, endpoint=True)
    fig.colorbar(cax, ticks=v)
    

    # Set up axes
    ax.set_xticklabels([''] + idx_to_class, rotation=90)
    ax.set_yticklabels([''] + idx_to_class)

    # Force label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    # sphinx_gallery_thumbnail_number = 2

    if outfile:
        plt.savefig(outfile)
        
    else:
        return fig
        
    # plt.savefig('confusion_matrix'+str(learning_values.pop())+'.jpg')
    # plt.show()


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)


def time_me(elapsedTime):

    hours = math.floor(elapsedTime / (60*60))
    elapsedTime = elapsedTime - hours * (60*60);
    minutes = math.floor(elapsedTime / 60)
    elapsedTime = elapsedTime - minutes * (60);
    seconds = math.floor(elapsedTime);

    return hours, minutes, seconds


if __name__ == '__main__':
    """ Main function """
    args = get_args()
    
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    # Run on GPU or CPU
    device = torch.device("cuda" if use_cuda else "cpu")

    # Fetch data
    dataloaders = data_loader.fetch_data(['train', 'test'], args)
    train_dataloader = dataloaders['train']
    test_dataloader = dataloaders['test']

    args.vocab_size += 1 
    print(args.output_size)

    idx_to_class, _ =  data_loader.get_classes(args.classses_file)

    # Load Model
    model = LocationRNN(args.vocab_size, args.output_size, args.embedding_dim, args.hidden_dim, args.n_layers)
    model = model.to(device)

    print(model)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Adam optimizer
    optimizer = optim.Adam(model.parameters(),
                           lr=args.learning_rate,
                           weight_decay=args.weight_decay)

    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    scheduler = optim.lr_scheduler.StepLR(optimizer,
                                          step_size=10,
                                          gamma=0.1)

    # print('Training for {} epoch(s)'.format(args.epochs))

    # time_start = time.time()

    # logging.info('Training Start')
    # #Train Model
    # for epoch in range(1, args.epochs + 1):
    #     scheduler.step()
    #     train(args, model, device, train_dataloader, optimizer, criterion, epoch, idx_to_class)

    #     if epoch % 10 == 0:
    #         print('Epoch: {}/{}'.format(epoch,args.epochs))

    #         save_checkpoint({
    #             'epoch': epoch,
    #             'arch': 'bidirectional_LSTM',
    #             'state_dict': model.state_dict(),
    #             'optimizer' : optimizer.state_dict(),
    #         })


    # time_end = time.time()
    # training_time = time_end - time_start 

    # print('Trainig Time: {}hr:{}min:{}s'.format(*time_me(training_time)))

    evaluate = True
    
    if evaluate: 
        test(args, model, device, test_dataloader, criterion, idx_to_class, 'predictions.csv')
    
    predict = False

    if predict:
        predict(args, model, device, test_dataloader, criterion, idx_to_class, 'predictions.csv')

    writer.close()