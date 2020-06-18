# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import pandas as pd
import numpy as np
import torch
import dgl
import random

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================


def from_csv(path, toolkit="rdkit", smiles_col=-1, y_cols=[-2], seed=2666, scale=1.0,
        **kwargs):
    """ Read csv from file.
    """

    def _from_csv():
        df = pd.read_csv(
            path,
            error_bad_lines=False,
            **kwargs)

        df = df.dropna(axis=0, subset=[df.columns[y_cols[0]]])

        df_smiles = df.iloc[:, smiles_col]
        df_y = df.iloc[:, y_cols]

        if toolkit == "rdkit":
            from rdkit import Chem

            df_smiles = [str(x) for x in df_smiles]

            idxs = [idx for idx in range(len(df_smiles)
                ) if 'nan' not in df_smiles[idx]]


            df_smiles = [df_smiles[idx] for idx in idxs]

            mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]
            gs = [pinot.graph.from_rdkit_mol(mol) for mol in mols]

        elif toolkit == "openeye":
            from openeye import oechem

            mols = [
                oechem.OESmilesToMol(oechem.OEGraphMol(), smiles)
                for smiles in df_smiles
            ]
            gs = [pinot.graph.from_oemol(mol) for mol in mols]

        ds = list(zip(gs, list(torch.tensor(scale * df_y.values[idxs], dtype=torch.float32))))
        random.seed(seed)
        random.shuffle(ds)

        return ds

    return _from_csv


def load_unlabeled_data(path, size=0.1, toolkit="rdkit", seed=2666):
    """ Read from unlabeled data set as background data
    """
    def _from_txt():
        f = open(path, "r")
        df_smiles = [line.rstrip() for line in f]
        # Since loading the whole data set takes a lot of time
        # Load only a subset of the data instead
        num_mols = int(len(df_smiles) * size)
        df_smiles = df_smiles[:num_mols]
        # Create "fake" labels
        df_y = torch.FloatTensor([1 for _ in range(num_mols)])

        if toolkit == "rdkit":
            from rdkit import Chem
            mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]
            gs = [pinot.graph.from_rdkit_mol(mol) for mol in mols]

        elif toolkit == "openeye":
            from openeye import oechem
            mols = [
                oechem.OESmilesToMol(oechem.OEGraphMol(), smiles)
                for smiles in df_smiles
            ]
            gs = [pinot.graph.from_oemol(mol) for mol in mols]

        gs = list(zip(gs, df_y))
        random.seed(seed)
        random.shuffle(gs)

        return gs

    return _from_txt

def normalize(ds):
    """ Get mean and std.
    """

    gs, ys = tuple(zip(*ds))
    y_mean = np.mean(ys)
    y_std = np.std(ys)

    def norm(y):
        return (y - y_mean) / y_std

    def unnorm(y):
        return y * y_std + y_mean

    return y_mean, y_std, norm, unnorm


def split(ds, partition):
    """ Split the dataset according to some partition.
    """
    n_data = len(ds)

    # get the actual size of partition
    partition = [int(n_data * x / sum(partition)) for x in partition]

    ds_batched = []
    idx = 0
    for p_size in partition:
        ds_batched.append(ds[idx : idx + p_size])
        idx += p_size

    return ds_batched


def batch(ds, batch_size, seed=2666):
    """ Batch graphs and values after shuffling.
    """
    # get the numebr of data
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    random.seed(seed)
    random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        torch.stack(ys[idx * batch_size : (idx + 1) * batch_size], dim=0)
        for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def batch_semi_supervised(ds, batch_size, seed=2666):
    """ Very similar to batch, but should be used when the 
    data contains None (data for semi-supervised training)
    because torch.tensor(arr) throws an error when arr
    contains any None
    """
    n_data_points = len(ds)
    n_batches = n_data_points // batch_size  # drop the rest

    np.random.seed(seed)
    np.random.shuffle(ds)
    gs, ys = tuple(zip(*ds))

    gs_batched = [
        dgl.batch(gs[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    ys_batched = [
        list(ys[idx * batch_size : (idx + 1) * batch_size])
        for idx in range(n_batches)
    ]

    return list(zip(gs_batched, ys_batched))


def prepare_semi_supervised_training_data(unlabelled_data, labelled_data, batch_size=32):
    """ Create a semi-supervised data by mixing unlabelled and labelled
    data. An example of labelled data can be readily accessed via
    `pinot.data.zinc_tiny()`
    """
    semi_supervised_data = []
    
    for (g, y) in unlabelled_data:
        semi_supervised_data.append((g, None))
    for (g, y) in labelled_data:
        semi_supervised_data.append((g, y))
        
    semi_supervised_data = batch_semi_supervised(semi_supervised_data, batch_size)
    return semi_supervised_data


def prepare_semi_supeprvised_data_from_labelled_data(labelled_data, r=0.2, seed=2666):
    """ Create semi-supervised data from labelled data by randomly removing the
    labels for (1-r) of the data points. 

    Returns:
        semi_data: the semi-supervised data with the same number of
            data points as labelled_data. However, only a fraction r
            of the points have labels

        small_labelled_data: the fraction r of the points with labels
    """
    semi_data = []
    small_labelled_data = []
    
    np.random.seed(seed)
    for (g,y) in labelled_data:
        if np.random.rand() < r:
            semi_data.append((g, y))
            small_labelled_data.append((g,y))
        else:
            semi_data.append((g, None))
    return semi_data, small_labelled_data