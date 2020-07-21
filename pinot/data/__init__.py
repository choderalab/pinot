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
    y_cols=[5, 6, 7, 8, 9, 10],
    time_col=-4,
    scale=0.01,
)

moonshot_sorted = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot_with_date.csv",
    smiles_col=1,
    y_cols=[8],
    scale=0.01,
    shuffle=False,
    dropna=True,
)
