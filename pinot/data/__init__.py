import pinot
from . import datasets
from . import utils
import os
import numpy as np
import sys


from .datasets import esol, freesolv, lipophilicity, curve, moonshot_mixed
from .unlabeled_datasets import moses_tiny, zinc_tiny,\
    moonshot_semi_small, esol_semi_small,\
    moonshot_unlabeled_small, esol_unlabeled_small

# =============================================================================
# PRESETS
# =============================================================================

covid = utils.from_csv(
    os.path.dirname(utils.__file__) + "/covid.tsv",
    smiles_col=7,
    y_cols=[10],
    delimiter="\t",
    dtype={"Smiles": str, "Standard Value": np.float32},
    header=1,
)


moonshot = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot.csv",
    smiles_col=0,
    y_cols=[6],
    scale=0.01,
    dropna=True,
)

moonshot_meta = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot.csv",
    smiles_col=0,
    y_cols=[3, 4, 5, 6, 7, 8, 9, 10, 11],
    scale=0.01,
)

moonshot_with_date = lambda: datasets.TemporalDataset().from_csv(
    os.path.dirname(utils.__file__) + "/moonshot_with_date.csv",
    smiles_col=1,
    y_cols=[11, 12, 13, 14, 15, 16],
    time_col=-2,
    scale=0.01,
)

moonshot_sorted = lambda: datasets.TemporalDataset().from_csv(
    os.path.dirname(utils.__file__) + "/moonshot_with_date.csv",
    smiles_col=1,
    y_cols=[14], # f inhibition at 20 uM
    scale=0.01,
    time_col=-2,
    dropna=True,
)

moonshot_multi = lambda: datasets.Dataset().from_csv(
    os.path.dirname(utils.__file__) + '/moonshot_[1-3-21].csv',
    smiles_col=1,
    y_cols=[11, 12, 13, 14, 15, 16],
    scale=0.01
)

moonshot_pic50 = lambda seed=None, shuffle=True: datasets.Dataset().from_csv(
    os.path.dirname(utils.__file__) + "/moonshot_IC50_filt.csv",
    smiles_col=3,
    y_cols=[-1],
    delimiter=',',
    dtype={"Smiles": str, "Standard Value": np.float32},
    header=1,
    seed=seed,
    shuffle=shuffle,
)