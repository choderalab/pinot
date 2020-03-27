import pinot.data.utils as utils
import os

esol = utils.from_csv(
    os.path.dirname(
        utils.__file__) + '/esol.csv')

