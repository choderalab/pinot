import pinot
from . import datasets
from . import utils
import os
import numpy as np
import sys



esol = utils.from_csv(os.path.dirname(utils.__file__) + "/esol.csv")
freesolv = utils.from_csv(os.path.dirname(utils.__file__) + "/SAMPL.csv", smiles_col=1, y_cols=[2])
lipophilicity = utils.from_csv(os.path.dirname(utils.__file__) + "/Lipophilicity.csv")



zinc_tiny = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/zinc/all.txt", size=0.01)
moses_tiny = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/moses/all.txt", size=0.01)

zinc_small = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/zinc/all.txt", size=0.1)
moses_small = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/moses/all.txt", size=0.1)

zinc_full = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/zinc/all.txt", size=1)
moses_full = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/moses/all.txt", size=1)

covid = utils.from_csv(
    os.path.dirname(utils.__file__) + "/covid.tsv",
    smiles_col=7,
    y_cols = [10],
    delimiter="\t",
    dtype={
        'Smiles': str,
        'Standard Value': np.float32
    },
    header=1)


moonshot = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot.csv",
    smiles_col=0,
    y_cols=[6],
    scale=0.01,
    dropna=True
)

moonshot_meta = utils.from_csv(
    os.path.dirname(utils.__file__) + "/moonshot.csv",
    smiles_col=0,
    y_cols=[3, 4, 5, 6, 7, 8, 9, 10, 11],

    scale=0.01,
)

moonshot_with_date = lambda: datasets.TemporalDataset(
        ).from_csv(
            os.path.dirname(utils.__file__) + '/moonshot_with_date.csv',
            smiles_col=1,
            y_cols=[5, 6, 7, 8, 9, 10],
            time_col=-3,
            scale=0.01)

moonshot_sorted = utils.from_csv(
        os.path.dirname(utils.__file__) + '/moonshot_with_date.csv',
        smiles_col=1,
        y_cols=[8],
        scale=0.01,
        shuffle=False,
        dropna=True)


def load_moonshot_semi_supervised(unlabeled_size=0.1, seed=2666):
    def load():
        moonshot_labeled = moonshot() # Get labeled and unlabeled data
        moonshot_unlabeled = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/moonshot_activity_synthetic.txt", unlabeled_size, seed=seed)()
        np.random.seed(seed)
        moonshot_labeled.extend(moonshot_unlabeled)
        np.random.shuffle(moonshot_labeled) # Combine and mix labeled and unlabeled data
        return moonshot_labeled
    return load

# This data set contains around 6200 molecules, 530 are labeled
moonshot_semi_small = load_moonshot_semi_supervised(0.1)
moonshot_unlabeled_small = utils.load_unlabeled_data(os.path.dirname(utils.__file__) + "/moonshot_activity_synthetic.txt", size=0.1)

# This data set contains around 60k molecules, 530 are labeled
moonshot_semi_all   = load_moonshot_semi_supervised(1.0)

def get_esol_semi():
    esol_semi, _ = utils.prepare_semi_supeprvised_data_from_labeled_data(esol(), r=0.5)
    return esol_semi

esol_semi = get_esol_semi
