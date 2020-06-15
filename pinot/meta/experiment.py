#=============================================================================
# IMPORTS
# =============================================================================
import torch
import pinot
import abc
import math

def get_separate_dataset(ds):
    n_assay = ds[0][1].shape[-1]
    print(n_assay)


get_separate_dataset(pinot.data.moonshot_meta())
