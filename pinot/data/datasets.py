# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import pandas as pd
import abc

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Dataset(abc.ABC, torch.utils.data.Dataset):
    """ The base class of map-style dataset.

    Parameters
    ----------
    graphs : list
        objects in the dataset

    Note
    ----
    This also supports iterative-style dataset by deleting `__getitem__`
    and `__len__` function.


    """
    def __init__(self, data_points=None):
        super(Dataset, self).__init__()
        self.data_points = data_points

    def __len__(self):
        # 0 len if no graphs
        if self.graphs is None:
            return 0

        else:
            return len(self.graphs)

    def __getitem__(self, idx):
        if self.graphs is None:
            raise RuntimeError('Empty molecule dataset.')

        if isinstance(idx, int): # sinlge element
            return self.graphs[idx]


        elif isinstance(idx, slice): # implement slicing
            # return a Dataset object rather than list
            return self.__class__(graphs=self.graphs[idx])


    def __iter__(self):

        # TODO:
        # is this efficient?
        graphs = iter(self.graphs)
        return graphs

    def split(self, partition):
        """ Split the dataset according to some partition.

        Parameters
        ----------
        partition : sequence of integers or floats

        """
        n_data = len(self)
        partition = [int(n_data * x / sum(partition)) for x in partition]
        ds = []
        idx = 0
        for p_size in partition:
            ds.append(self[idx : idx + p_size])
            idx += p_size

        return ds

    def save(self, path):
        """ Save dataset to path.

        Parameters
        ----------
        path : path-like object
        """
        import pickle
        with open(path, 'wb') as f_handle:
            pickle.dump(
                    self.graphs,
                    f_handle)

    def load(self, path):
        """ Load path to dataset.

        Parameters
        ----------
        """
        import pickle
        with open(path, 'rb') as f_handle:
            self.graphs = pickle.load(f_handle)

    def from_csv(self, path):
