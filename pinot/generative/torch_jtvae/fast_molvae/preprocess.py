import torch
import torch.nn as nn
from multiprocessing import Pool

import math, random, sys
import argparse
import pickle
import numpy as np

from pinot.generative.torch_jtvae.fast_jtnn import *
import rdkit
import time
import os

def tensorize(smiles, assm=True):
    mol_tree = MolTree(smiles)
    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree


# Print iterations progress
def printProgressBar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()


if __name__ == "__main__":
    lg = rdkit.RDLogger.logger() 
    lg.setLevel(rdkit.RDLogger.CRITICAL)

    parser = argparse.ArgumentParser("Preprocessing script to parse SMILES into molecular junction trees")
    parser.add_argument("-t", "--train", help="Path to the file containing SMILES for training")
    parser.add_argument("-n", "--split", default=100, help="Number of sub-data splits, use a large number if you want to experiment with smaller sub-data")
    parser.add_argument("--out_folder", default="jt_vae_parsed_molecular_trees", help="Folder to save the parsed molecular trees")

    args = parser.parse_args()

    num_splits = int(args.split)

    with open(args.train) as f:
        data = [line.strip("\r\n ").split()[0] for line in f]

    # Create an output folder if it does not already exist
    if not os.path.isdir(args.out_folder):
        os.makedirs(args.out_folder)
    
    print("Finished reading in", len(data), "molecules")
    start = time.time()
    all_data = []
    N = len(data)
    for i, smile in enumerate(data):
        all_data.append(tensorize(smile))
        if i % 100 == 0:
            printProgressBar(i+1, N, "Processed", "of {} molecules".format(N))    

    print("Finished processing all the data in {} seconds! Now splitting it".format(time.time()-start))
    
    le = int((len(all_data) + num_splits - 1) / num_splits)

    for split_id in np.random.permutation(num_splits):
        st = split_id * le
        sub_data = all_data[st : min(st + le, len(all_data))]

        with open(os.path.join(args.out_folder, 'tensors-%d.pkl' % split_id), 'wb') as f:
            torch.save(sub_data, f)

