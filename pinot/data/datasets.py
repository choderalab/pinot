# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import pandas as pd
import abc
import torch

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
    def __init__(self, ds=None):
        super(Dataset, self).__init__()
        self.ds = ds

    def __len__(self):
        # 0 len if no graphs
        if self.ds is None:
            return 0

        else:
            return len(self.graphs)

    def __getitem__(self, idx):
        if self.ds is None:
            raise RuntimeError('Empty molecule dataset.')

        if isinstance(idx, int): # sinlge element
            return self.ds[idx]

        elif isinstance(idx, slice): # implement slicing
            # return a Dataset object rather than list
            return self.__class__(ds=self.ds[idx])

    def __iter__(self):

        # TODO:
        # is this efficient?
        ds = iter(self.ds)
        return ds

    def split(self, *args, **kwargs):
        """ Split the dataset according to some partition.

        Parameters
        ----------
        partition : sequence of integers or floats

        """
        ds = piont.data.utils.split(self, *args, **kwargs)
        return ds

    def batch(self, *args, **kwargs):
        ds = pinot.data.utils.batch(self, *args, **kwargs)
        return ds

    def from_csv(self, *args, **kwargs):
        self.ds = pinot.data.utils.from_csv(*args, **kwargs)()
        return self


class TemporalDataset(Dataset):
    """ Dataset with time.

    """
    def __init__(self, ds):
        super(TemporalDataset, self).__init__(ds)


    def from_csv(
        self,
        path,
        toolkit="rdkit",
        smiles_col=-1,
        time_col=-3,
        y_cols=[-2],
        seed=2666,
        scale=1.0,
        dropna=False,
        **kwargs
    ):
        """ Read csv from file.
        """

        def _from_csv():
            df = pd.read_csv(path, error_bad_lines=False, **kwargs)

            if dropna is True:
                df = df.dropna(axis=0, subset=[df.columns[y_cols[0]]])

            df_smiles = df.iloc[:, smiles_col]
            df_y = df.iloc[:, y_cols]
            df_time = df.iloc[:, time_col]

            if toolkit == "rdkit":
                from rdkit import Chem

                df_smiles = [str(x) for x in df_smiles]

                idxs = [
                    idx
                    for idx in range(len(df_smiles))
                    if "nan" not in df_smiles[idx]
                ]

                df_smiles = [df_smiles[idx] for idx in idxs]

                mols = [Chem.MolFromSmiles(smiles) for smiles in df_smiles]
                gs = [pinot.graph.from_rdkit_mol(mol) for mol in mols]

            elif toolkit == "openeye":
                raise NotImplementedError


            ds = list(
                zip(
                    gs,
                    list(
                        torch.tensor(
                            scale * df_y.values[idxs], dtype=torch.float32
                        )
                    ),
                   list(df_time.values[idxs]), 
                )
            )

            return ds

        return _from_csv


