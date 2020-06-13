import pinot
import pinot.data
from . import utils
import os

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
    os.path.dirname(utils.__file__) + "/covid.csv",
    smiles_col=7,
    y_cols = [10])
