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
    """ The base class of map-style dataset."""

    def __init__(self, ds=None):
        super(Dataset, self).__init__()
        self.ds = ds

    def __len__(self):
        # 0 len if no graphs
        if self.ds is None:
            return 0

        else:
            return len(self.ds)

    def __getitem__(self, idx):
        if self.ds is None:
            raise RuntimeError("Empty molecule dataset.")

        if isinstance(idx, int):  # sinlge element
            return self.ds[idx]

        elif isinstance(idx, slice):  # implement slicing
            # return a Dataset object rather than list
            return self.__class__(ds=self.ds[idx])

    def __iter__(self):

        # TODO:
        # is this efficient?
        ds = iter(self.ds)
        return ds

    def split(self, *args, **kwargs):
        """Split the dataset according to some partition. """
        ds = piont.data.utils.split(self, *args, **kwargs)
        return ds

    def batch(self, *args, **kwargs):
        """Batch dataset."""
        ds = pinot.data.utils.batch(self, *args, **kwargs)
        return ds

    def from_csv(self, *args, **kwargs):
        """Read csv dataset. """
        self.ds = pinot.data.utils.from_csv(*args, **kwargs)()
        return self


class TemporalDataset(Dataset):
    """ Dataset with time.

    Methods
    -------
    from_csv : Read data from csv.

    split_by_time : Split dataset by a certain date.

    filter_by_time : Filter the data by a certain date.

    """

    def __init__(self, ds=None):
        super(TemporalDataset, self).__init__(ds)

    def from_csv(
        self,
        path,
        smiles_col,
        y_cols,
        time_col,
        seed=2666,
        scale=1.0,
        dropna=False,
        toolkit="rdkit",
    ):
        """Read csv from file.

        Parameters
        ----------
        path : `str`
            Path to the csv file.

        smiles_col : `int`
            The column with SMILES strings.

        y_cols : `List` of `int`
            The columns with SMILES strings.

        time_col : `int`
            The column with time.

        scale : `float`
             (Default value = 1.0)
             Scaling the input.

        dropna : `bool`
             (Default value = False)
             Whether to drop `NaN` values in the column.


        toolkit : `str`. `rdkit` or `openeye`
             (Default value = "rdkit")
             Toolkit used to read molecules.


        """

        def _from_csv():
            """ """
            df = pd.read_csv(path, error_bad_lines=False)

            df = df.sort_values(by=df.columns[time_col], ascending=True)

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

        ds = _from_csv()
        self.ds = ds
        return self

    def split_by_time(self, time):
        """ Split dataset by a certain date. """
        before = []
        after = []

        for g, y, t in self.ds:
            if t < time:
                before.append((g, y))

            if t >= time:
                after.append((g, y))

        before = Dataset(before)
        after = Dataset(after)

        return before, after

    def filter_by_time(self, after="1989-06-04", before="2666-12-31"):
        """ Filter the data by a certain date.

        Parameters
        ----------
        after : `str`
             (Default value = '1989-06-04')


        before : `str`
             (Default value = '2666-12-31')

        Returns
        -------
        between : `List` of `(dgl.DGLGraph, torch.Tensor)`

        """
        between = []
        for g, y, t in self.ds:
            if after <= t <= before:
                between.append((g, y))

        return between
