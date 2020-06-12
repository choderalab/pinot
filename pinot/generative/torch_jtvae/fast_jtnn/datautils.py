import torch
from torch.utils.data import Dataset, DataLoader
from pinot.generative.torch_jtvae.fast_jtnn.mol_tree import MolTree
import numpy as np
from pinot.generative.torch_jtvae.fast_jtnn.jtnn_enc import JTNNEncoder
from pinot.generative.torch_jtvae.fast_jtnn.mpn import MPN
from pinot.generative.torch_jtvae.fast_jtnn.jtmpn import JTMPN
import pickle
import os, random

class PairTreeFolder(object):

    def __init__(self, data_folder, vocab, batch_size, num_workers=4, shuffle=True, y_assm=True, replicate=None):
        self.data_folder = data_folder
        self.data_files = [fn for fn in os.listdir(data_folder)]
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.y_assm = y_assm
        self.shuffle = shuffle

        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

    def __iter__(self):
        for fn in self.data_files:
            fn = os.path.join(self.data_folder, fn)
            with open(fn) as f:
                data = pickle.load(f)

            if self.shuffle: 
                random.shuffle(data) #shuffle data before batch

            batches = [data[i : i + self.batch_size] for i in range(0, len(data), self.batch_size)]
            if len(batches[-1]) < self.batch_size:
                batches.pop()

            dataset = PairTreeDataset(batches, self.vocab, self.y_assm)
            dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

            for b in dataloader:
                yield b

            del data, batches, dataset, dataloader

class MolTreeFolder(object):
    def __init__(self, data_file, vocab, batch_size, num_workers=4, shuffle=True, assm=True, replicate=None):
        self.data_file = data_file
        self.batch_size = batch_size
        self.vocab = vocab
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.assm = assm
        
        ################## Construct a data loader ###################
        if replicate is not None: #expand is int
            self.data_files = self.data_files * replicate

        data = torch.load(self.data_file)

        # if self.shuffle: 
        #     random.shuffle(data) #shuffle data before batch
        # Get the trees and labels
        trees, y = tuple(zip(*data))

        tree_batches = [trees[i : i + self.batch_size] for i in range(0, len(trees), self.batch_size)]
        # if len(tree_batches[-1]) < self.batch_size:
        #     batches.pop()

        y_batches = [y[i : i + self.batch_size] for i in range(0, len(y), self.batch_size)]


        dataset = MolTreeDataset(tree_batches, y_batches, self.vocab, self.assm)
        self.dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=self.num_workers, collate_fn=lambda x:x[0])

    def getDataLoader(self):
        return self.dataloader

class PairTreeDataset(Dataset):

    def __init__(self, data, vocab, y_assm):
        self.data = data
        self.vocab = vocab
        self.y_assm = y_assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        batch0, batch1 = zip(*self.data[idx])
        return tensorize(batch0, self.vocab, assm=False), tensorize(batch1, self.vocab, assm=self.y_assm)

class MolTreeDataset(Dataset):

    def __init__(self, data, label, vocab, assm=True):
        self.data = data
        self.label = label
        self.vocab = vocab
        self.assm = assm

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return (tensorize(self.data[idx], self.vocab, assm=self.assm), self.label[idx])

def tensorize(tree_batch, vocab, assm=True):
    set_batch_nodeID(tree_batch, vocab)
    smiles_batch = [tree.smiles for tree in tree_batch]
    jtenc_holder,mess_dict = JTNNEncoder.tensorize(tree_batch)
    jtenc_holder = jtenc_holder
    mpn_holder = MPN.tensorize(smiles_batch)

    if assm is False:
        return tree_batch, jtenc_holder, mpn_holder

    cands = []
    batch_idx = []
    for i,mol_tree in enumerate(tree_batch):
        for node in mol_tree.nodes:
            #Leaf node's attachment is determined by neighboring node's attachment
            if node.is_leaf or len(node.cands) == 1: continue
            cands.extend( [(cand, mol_tree.nodes, node) for cand in node.cands] )
            batch_idx.extend([i] * len(node.cands))

    jtmpn_holder = JTMPN.tensorize(cands, mess_dict)
    batch_idx = torch.LongTensor(batch_idx)

    return tree_batch, jtenc_holder, mpn_holder, (jtmpn_holder,batch_idx)

def set_batch_nodeID(mol_batch, vocab):
    tot = 0
    for mol_tree in mol_batch:
        for node in mol_tree.nodes:
            node.idx = tot
            node.wid = vocab.get_index(node.smiles)
            tot += 1
