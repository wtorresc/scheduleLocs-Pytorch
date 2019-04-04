import numpy as np
import pandas as pd
import random
import torch
import io
import os
from torch.utils.data import Dataset, DataLoader
import json
import pdb
import itertools as it
from collections import Counter, OrderedDict
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


# def get_classes(labels_file):
#     """
#     Finds the class folders in a dataset.
#     Args:
#        dir (string): Root directory path.
#     Returns:
#        tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
#     Ensures:
#        No class is a subdirectory of another.
#     """

#     with open(labels_file) as f:
#         data = json.load(f, object_pairs_hook=OrderedDict)
#         data_dict = data['uniformat']
#         fieldnames = ('level_1','level_2','level_3', 'level_4','level_5')
#         labels = np.empty((len(data_dict), len(fieldnames)), dtype='<U8')
#         for idx, _ in enumerate(data_dict):
#             labels[idx] = list(data_dict[idx].values())

#     Empty = '0'
#     classes = []
#     class_to_idx = []

#     for i in range(len(fieldnames)):
#         classes.append(sorted(np.unique(labels[:,i]).tolist()))
#         classes[i].insert(0,Empty) #add a NONE class
#         class_to_idx.append({classes[i][j]: j for j in range(len(classes[i]))})

#     return classes, class_to_idx


class LocationDataset(Dataset):

    # def __init__(self, x_data_to_process, y_data_to_process, seq_length):
    def __init__(self, seq_length, vocab_size, train=True):
        """ inputs for x and y values are given as a json file """
        path='./data/sch2json.json'
        path_test = './data/sch2json_test.json'
        labels_path = './data/uniformat.json'
        format='json'

        x_train = []
        y_train = []
        NumLevels = 5
        Empty = '0'

        with io.open(os.path.expanduser(path), encoding="utf8") as f:
          reader = f
          for line in reader:
            line = json.loads(line, object_pairs_hook = OrderedDict)

            x_train.append(line['activity'])
            y_train.append(line['target'])

        # if not train:

        #     x_test = []
        #     y_test = []

        #     with io.open(os.path.expanduser(path_test), encoding="utf8") as f:
        #       reader = f
        #       for line in reader:
        #         line = json.loads(line, object_pairs_hook = OrderedDict)

        #         x_test.append(nltk.word_tokenize(line['activity']))
        #         y_test.append(line['target'])
            
        classes, class_to_idx = self._get_classes(y_train)
        # classes, class_to_idx = get_classes(labels_path)

        self.classes_size = len(classes)

        dataset_size = len(y_train)

        # #shuffle
        # random_idx = list(range(dataset_size))#, dtype=np.int32)
        # random.Random(4).shuffle(random_idx)
        # x_train = [x_train[idx] for idx in random_idx]
        # y_train = [y_train[idx] for idx in random_idx]

        train_split = 0.80
        split = int(np.floor(train_split * dataset_size))

        if not train:
            y_test = y_train[split:]
            x_test = x_train[split:]

        y_train = y_train[:split]
        x_train = x_train[:split]
        labels = y_train

        if not train:
            labels = y_test 

        ## Encode the labels with the classes given
        encoded_labels = []
        
        for y_i in labels:
            encoded_label_i = []
            for y_ii in y_i:
                if y_ii != None:
                    encoded_label_i.append(class_to_idx[str(y_ii)])
                else:
                    encoded_label_i.append(class_to_idx['0'])
            encoded_labels.append(encoded_label_i)

        ### word_to_id and id_to_word. associate an id to every unique token in the training data
        all_tokens = it.chain.from_iterable(x_train)
        
        ## Build a dictionary that maps words to integers based on training data
        counts = Counter(all_tokens)
        vocab = sorted(counts, key=counts.get, reverse=True)
        word_to_id = {token:idx for idx, token in enumerate(vocab, 1)}
        word_to_id["<UNK>"] = 0
        # word_to_id["<START>"] = 1
        
        id_to_word = {idx:word for word,idx in word_to_id.items()}# Find a clever way to fix this!!!!

        # pdb.set_trace()
        ## assign 0 if token doesn't appear in our dictionary
        ## we want to reserve id=0 for an unknown token
        ## word_to_idx is only based on words seeing on training data
        if train:
            x_train_token_ids = [[word_to_id.get(token,0) for token in x] for x in x_train]
            x_token_ids = x_train_token_ids
        else:
            x_test_token_ids = [[word_to_id.get(token,0) for token in x] for x in x_test]
            x_token_ids = x_test_token_ids

        # x_test_token_ids = [[word_to_id.get(token,0) for token in x] for x in x_test]

        print('Number of reviews before removing outliers: ', len(x_token_ids))

        ## remove any activity/label with zero length from the reviews_ints list.

        # get indices of any reviews with length 0
        non_zero_idx = [idx for idx, activity in enumerate(x_token_ids) if len(activity) != 0]

        # remove 0-length reviews and their labels
        x_token_ids = [torch.LongTensor(x_token_ids[idx]) for idx in non_zero_idx]
        encoded_labels = [torch.LongTensor(encoded_labels[ii]) for ii in non_zero_idx]

        print('Number of reviews after removing outliers: ', len(x_token_ids))

        # Test your implementation!

        # seq_length = 15
        print('seq_length: {}'.format(seq_length))
        seq_lengths = torch.LongTensor(list(map(len, x_token_ids)))

        '''
        Remember that you can manipulate the length of the sequences seq_legths[seq_legths>seq_legth] = seq_legth 
        '''
        # ordered_tensor  = sorted(x_train_token_ids, key=len, reverse=True)
        # padded_tensor = torch.nn.utils.rnn.pad_sequence(ordered_tensor, batch_first=True)
        padded_tensor = torch.nn.utils.rnn.pad_sequence(x_token_ids, batch_first=True)
        padded_labels = torch.nn.utils.rnn.pad_sequence(encoded_labels, batch_first=True)

        # Set to zero indices above the vocab size
        padded_tensor[padded_tensor > vocab_size] = 0


        # pdb.set_trace()
        # features = self.pad_features(x_train_token_ids, seq_length)

        ## test statements - do not change - ##
        assert len(padded_tensor)==len(x_token_ids), "Your features should have as many rows as reviews."
        assert len(padded_tensor)==len(padded_labels), "Your features should have as many rows as reviews."
        '''
        Save this assertion for when you control the length of the sequences
        '''      
        # assert len(padded_tensor[0])==seq_length, "Each feature row should contain seq_length values."
        self.len = len(x_token_ids)

        self.features = padded_tensor #features
        self.encoded_labels = padded_labels #encoded_labels
        self.seq_lengths = seq_lengths

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """      
        sample = self.features[index]
        target = self.encoded_labels[index]
        seq_lengths = self.seq_lengths[index]

        return sample, target, seq_lengths

    def __len__(self):
        return self.len

    def pad_features(self, activities_ints, seq_length):
        ''' Return features of activities_ints, where each activity is padded with 0's 
            or truncated to the input seq_length.
        '''
        
        # getting the correct rows x cols shape
        features = np.zeros((len(activities_ints), seq_length), dtype=int)

        # for each activity, I grab that activity and 
        for i, row in enumerate(activities_ints):
          start_index = 0
          if len(row) > seq_length:
            start_index = np.random.randint(len(row) - seq_length + 1)  

          features[i, -len(row):] = np.array(row)[start_index : start_index + seq_length]
        
        return features

    def _get_classes(self, labels):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        classes = list(set(it.chain(*labels)))
        classes = [str(i) for i in classes]
        classes.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))
        class_to_idx = {classes[idx]: idx for idx in range(len(classes))}

        return classes, class_to_idx

def get_traindata(args):
    trainset = LocationDataset(#data_dir = args.data_dir,
                                seq_length = 12,
                                vocab_size = args.vocab_size,
                                train = True                                
                                )
    return trainset

def get_testdata(args):
    testset = LocationDataset(#data_dir = args.data_dir,
                                seq_length = 12,
                                vocab_size = args.vocab_size,
                                train = False                                
                                )
    return testset

def fetch_data(types, args):
    """
    Fetches the DataLoader object for each type in types from data_dir.
    Args:
        types: (list) 'train' and'test' datasets
        args: (args) folder containing the datasets and number of workers
    Return
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'test']:
        if split in types:
            path = os.path.join(args.data_dir)

            if split == 'train':
                trainset = get_traindata(args)
                args.output_size = trainset.classes_size
                dataloaded = DataLoader(trainset,
                                batch_size = args.batchsize,
                                shuffle = True,
                                num_workers = args.num_workers,
                                drop_last = True
                                )
            else:
                testset = get_testdata(args)
                dataloaded = DataLoader(testset,
                                batch_size = args.test_batchsize,
                                shuffle = False,
                                num_workers = args.num_workers,
                                drop_last = False
                                )

            dataloaders[split] = dataloaded

    return dataloaders


if __name__ == '__main__':
    data = LocationDataset(8, 1000)
#     # print(data.__getitem__(0))
#     # for i in range(100):
#     #   print(data.__getitem__(i))

#     dataloaded = DataLoader(data,
#                     batch_size=16,
#                     shuffle = True,
#                     num_workers =4
#                     )

#     pdb.set_trace()

