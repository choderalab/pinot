# =============================================================================
# IMPORTS
# =============================================================================
import pinot
import numpy as np
import dgl
import pandas as pd
import abc
import torch
import os
from pinot.data import utils
from rdkit import Chem

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
        ds = pinot.data.utils.split(self, *args, **kwargs)
        return ds

    def batch(self, *args, **kwargs):
        """Batch dataset."""
        ds = pinot.data.utils.batch(self, *args, **kwargs)
        return ds

    def shuffle(self, *args, **kwargs):
        """ Shuffle the records in the dataset. """
        import random

        random.shuffle(self.ds)
        return self

    def from_csv(self, *args, **kwargs):
        """Read csv dataset. """
        self.ds = pinot.data.utils.from_csv(*args, **kwargs)()
        return self


class AttributedDataset(Dataset):
    """ Dataset with attributes. """

    def __init__(self, ds=None):
        super(AttributedDataset, self).__init__()
        self.ds = ds

    def from_csv(
        self,
        path,
        smiles_col,
        y_cols,
        attr_cols,
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

        attr_cols : `int`
            The columns with attributes.

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

            if dropna is True:
                df = df.dropna(axis=0, subset=[df.columns[y_cols[0]]])

            df_smiles = df.iloc[:, smiles_col]
            df_y = df.iloc[:, y_cols]
            df_attr = df.iloc[:, attr_cols]

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
                    list(df_attr.values[idxs]),
                )
            )

            return ds

        ds = _from_csv()
        self.ds = ds
        return self


class NamedAttributedDataset(Dataset):
    """ Dataset with attributes. """

    def __init__(self, ds=None):
        super(NamedAttributedDataset, self).__init__()
        self.ds = ds

    def from_csv(
        self,
        path,
        smiles_col,
        y_cols,
        seed=2666,
        scale=1.0,
        dropna=False,
        toolkit="rdkit",
        **kwargs,
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

            if dropna is True:
                df = df.dropna(axis=0, subset=[df.columns[y_cols[0]]])

            df_smiles = df.iloc[:, smiles_col]
            df_y = df.iloc[:, y_cols]

            # initialize dataframes
            dfs = {}

            for attr_name, col in kwargs:
                if attr_name.endswith("col"):
                    assert isinstance(col, int)
                    dfs[attr_name] = df.iloc[:, col]

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
                    *[list(df.values[idxs]) for df in dfs],
                )
            )

            return ds

        ds = _from_csv()
        self.ds = ds
        self.attr_names = list(kwargs.keys())
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


class MixedSingleAndMultipleDataset(Dataset):
    """ Dataset object with Dictionary view. """

    def __init__(self, ds=None):
        super(MixedSingleAndMultipleDataset, self).__init__(ds)
        self._number_of_measurements = None

    @property
    def number_of_measurements(self):
        if self._number_of_measurements is None:
            raise RuntimeError('Only available after viewing all the pairs.')

        else:
            return self._number_of_measurements

    @property
    def number_of_unique_graphs(self):
        return len(self)


    def from_csv(
        self,
        # paths
        single_path,
        multiple_path,
        # simles column
        smiles_col_name="SMILES",
        # specification for multiple
        multiple_concentration_col_name="f_concentration_uM",
        multiple_measurement_col_name="f_inhibition_list",
        # specification for signel
        single_concentrations=[20, 50],
        single_col_names=["f_inhibition_at_20_uM", "f_inhibition_at_50_uM",],
    ):
        def _from_csv():
            # read single and multiple data
            df_single = pd.read_csv(single_path)
            df_multiple = pd.read_csv(multiple_path)

            # merge it
            df = df_single.merge(right=df_multiple, how="left", on="SMILES")

            # filter it
            df = df.filter(
                items=[
                    smiles_col_name,
                    multiple_concentration_col_name,
                    multiple_measurement_col_name,
                    *single_col_names,
                ]
            )

            df["cs_single"] = np.nan
            df["ys_single"] = np.nan

            df = df.rename(
                columns={
                    multiple_concentration_col_name: "cs_multiple",
                    multiple_measurement_col_name: "ys_multiple",
                }
            )

            def flatten_multiple(record):
                cs = record["cs_multiple"]
                ys = record["ys_multiple"]

                if isinstance(cs, str):
                    cs = eval(cs)
                    ys = eval(ys)

                    cs = [x for c in cs for x in c]
                    ys = [x for y in ys for x in y]

                    record["cs_multiple"] = cs
                    record["ys_multiple"] = ys

                return record

            df = df.apply(flatten_multiple, axis=1)

            def flatten_single(record):
                cs = single_concentrations
                ys = [record[name] for name in single_col_names]

                record["cs_single"] = cs
                record["ys_single"] = ys

                return record

            df = df.apply(flatten_single, axis=1)

            self.ds = df.to_dict(orient="records")

            for record in self.ds:
                record["g"] = pinot.graph.from_rdkit_mol(
                    Chem.MolFromSmiles(record["SMILES"])
                )

            # print(self.ds)

            return self

        return _from_csv

    @staticmethod
    def all_available_pairs(xs):
        # initialize return lists
        gs = []
        cs = []
        ys = []

        for x in xs:  # loop through the data
            # get the graph
            g = x["g"]

            # get the single point measurements
            ys_single = x["ys_single"]
            cs_single = x["cs_single"]

            # get the multiple point measurements
            ys_multiple = x["ys_multiple"]
            cs_multiple = x["cs_multiple"]

            # append the results
            if isinstance(cs_single, list) and isinstance(ys_single, list):
                for c, y in zip(cs_single, ys_single):
                    if ~np.isnan(c) and ~np.isnan(y):
                        gs.append(g)
                        cs.append(c)
                        ys.append(y)

            if isinstance(cs_multiple, list) and isinstance(ys_multiple, list):
                for c, y in zip(cs_multiple, ys_multiple):
                    if ~np.isnan(c) and ~np.isnan(y):
                        gs.append(g)
                        cs.append(c)
                        ys.append(y)

        # batch
        g = dgl.batch(gs)
        c = torch.tensor(cs)[:, None]
        y = torch.tensor(ys)[:, None]

        return g, c, y

    @staticmethod
    def all_graphs(xs):
        return dgl.batch([x["g"] for x in xs])

    @staticmethod
    def _rebatch(xs, *args, **kwargs):
        assert len(xs) == 1
        g, c, y = xs[0]
        gs = dgl.unbatch(g)
        cs = c.numpy().tolist()
        ys = y.numpy().tolist()

        _ds = list(zip(gs, cs, ys))

        def _collate_fn(_xs):
            _gs = []
            _cs = []
            _ys = []

            for _g, _c, _y in _xs:
                _gs.append(_g)
                _cs.append(_c)
                _ys.append(_y)

            _gs = dgl.batch(_gs)
            _cs = torch.tensor(_cs)
            _ys = torch.tensor(_ys)

            return _gs, _cs, _ys

        return torch.utils.data.DataLoader(
            dataset=_ds, collate_fn=_collate_fn, *args, **kwargs
        )

    def view(self, collate_fn="all_available_pairs", *args, **kwargs):
        """ View the dataset as loader. """
        if collate_fn == 'fixed_size_batch':
            _ds = [self.all_available_pairs(self.ds)]
            self._number_of_measurements = _ds[0][1].shape[0]
            return self._rebatch(_ds, *args, **kwargs)

        if collate_fn == "all_graphs":
            kwargs["batch_size"] = len(self)  # ensure all the graph

        if isinstance(collate_fn, str):
            collate_fn = getattr(self, collate_fn)

        return torch.utils.data.DataLoader(
            dataset=self, collate_fn=collate_fn, *args, **kwargs,
        )




# =============================================================================
# PRESETS
# =============================================================================
def esol():
    return Dataset().from_csv(os.path.dirname(utils.__file__) + "/esol.csv")


def lipophilicity():
    return Dataset().from_csv(
        os.path.dirname(utils.__file__) + "/SAMPL.csv",
        smiles_col=1,
        y_cols=[2],
    )


def freesolv():
    return Dataset().from_csv(
        os.path.dirname(utils.__file__) + "/SAMPL.csv",
        smiles_col=1,
        y_cols=[2],
    )


def curve():
    return AttributedDataset().from_csv(
        os.path.dirname(utils.__file__) + "/curve.csv",
        smiles_col=1,
        y_cols=[2],
        attr_cols=[3],
    )

def moonshot_mixed():
    return pinot.data.datasets.MixedSingleAndMultipleDataset().from_csv(
        os.path.dirname(utils.__file__) + "/activity_data.csv",
        os.path.dirname(utils.__file__)
        + "/fluorescence_df_for_chodera_lab.csv",
    )()
