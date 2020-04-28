import pinot
import pinot.data
import .utils
import os

esol = utils.from_csv(
    os.path.dirname(
        utils.__file__) + '/esol.csv')

