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

def make_tree(smiles, assm=True):
    mol_tree = MolTree(smiles)
    
    cset = set()
    for c in mol_tree.nodes:
        cset.add(c.smiles)

    mol_tree.recover()
    if assm:
        mol_tree.assemble()
        for node in mol_tree.nodes:
            if node.label not in node.cands:
                node.cands.append(node.label)

    del mol_tree.mol
    for node in mol_tree.nodes:
        del node.mol

    return mol_tree, cset


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
    parser.add_argument("--smiles", help="Path to the file containing SMILES for training")
    parser.add_argument("--trees_out", default="jt_trees.pkl", help="File to save the parsed molecular trees")
    parser.add_argument("--vocab_out", required=True, help="Output file for the parsed vocabulary")
    parser.add_argument("--labelled", action="store_true", default=False, help="Flag as labelled data set")

    args = parser.parse_args()

    smiles = []
    with open(args.smiles) as f:
        for line in f:
            l = line.rstrip()
            if args.labelled:
                smile, y = tuple(l.split(","))
                smiles.append((smile, y))
            else:
                smile = l
                smiles.append((smile, 0))
    
    print("Finished reading in", len(smiles), "molecules")
    start = time.time()
    trees = []
    N = len(smiles)
    wset  = set()
    for i, (smile,y) in enumerate(smiles):
        tree, cset = make_tree(smile)
        trees.append((tree, y)) # Construct junction trees
        wset.update(cset) # Add to the grow set of WORDS
        printProgressBar(i+1, N, "Processed", "of {} molecules".format(N))    

    print("Finished processing all the data in {} seconds!".format(time.time()-start))
    print("Now saving the parsed trees and vocabulary to specified files")
    with open(args.trees_out, 'wb') as f:
        torch.save(trees, f)
        f.close()

    with open(args.vocab_out, "w") as f:
        for x in wset:
            f.write(x)
            f.write("\n")
        f.close()

