import pinot
import numpy as np
from . import utils
import os

def zinc_tiny():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/zinc/all.txt", size=0.01
    )

def moses_tiny():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/moses/all.txt", size=0.01
    )

def zinc_small():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/zinc/all.txt", size=0.1
    )

def moses_small():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/moses/all.txt", size=0.1
    )

def zinc_full():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/zinc/all.txt", size=1
    )

def moses_full():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/moses/all.txt", size=1
    )


def load_moonshot_semi_supervised(unlabeled_size=0.1, seed=2666):
    """

    Parameters
    ----------
    unlabeled_size :
         (Default value = 0.1)
    seed :
         (Default value = 2666)

    Returns
    -------

    """

    moonshot_labeled = pinot.data.moonshot()  # Get labeled and unlabeled data

    moonshot_unlabeled = pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__)
        + "/moonshot_activity_synthetic.txt",
        unlabeled_size,
        seed=seed,
    )()

    np.random.seed(seed)
    moonshot_labeled.extend(moonshot_unlabeled)
    np.random.shuffle(
        moonshot_labeled
    )  # Combine and mix labeled and unlabeled data
    return moonshot_labeled

# This data set contains around 6200 molecules, 530 are labeled
def moonshot_semi_small():
    return load_moonshot_semi_supervised(0.1)

def moonshot_unlabeled_small():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/moonshot_activity_synthetic.txt",
        size=0.1,
    )

def moonshot_unlabeled_all():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/moonshot_activity_synthetic.txt",
        size=1.0,
    )

# This data set contains around 60k molecules, 530 are labeled
def moonshot_semi_all():
    return load_moonshot_semi_supervised(1.0)


def load_esol_semi_supervised(unlabeled_size=0.1, seed=2666):
    """

    Parameters
    ----------
    unlabeled_size :
         (Default value = 0.1)
    seed :
         (Default value = 2666)

    Returns
    -------

    """

    esol_labeled = pinot.data.esol()  # Get labeled and unlabeled data
    esol_unlabeled = utils.load_unlabeled_data(
        os.path.dirname(utils.__file__) + "/esol_synthetic_smiles.txt",
        unlabeled_size,
        seed=seed,
    )()
    np.random.seed(seed)
    esol_labeled.extend(esol_unlabeled)
    np.random.shuffle(
        esol_labeled
    )  # Combine and mix labeled and unlabeled data
    return esol_labeled

# This data set contains around 12000 molecules
def esol_semi_small():
    return load_esol_semi_supervised(0.1)

def esol_unlabeled_small():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/esol_synthetic_smiles.txt", size=0.1
    )

def esol_unlabeled_all():
    return pinot.data.datasets.UnlabeledDataset().from_txt(
        os.path.dirname(utils.__file__) + "/esol_synthetic_smiles.txt", size=1.0
    )

# This data set contains around 120000 molecules
def esol_semi_all():
    return load_esol_semi_supervised(1.0)
