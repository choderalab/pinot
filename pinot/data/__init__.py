import pinot
# import pinot.data
# import pinot.data.utils as utils
import os

import utils

esol = utils.from_csv(
    os.path.dirname(
        utils.__file__) + '/esol.csv')

