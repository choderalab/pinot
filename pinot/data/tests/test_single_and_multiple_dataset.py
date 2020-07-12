import pytest


def test_mix():
    import pinot
    from pinot.data import utils
    import os

    df = pinot.data.datasets.MixedSingleAndMultipleDataset().from_csv(
        os.path.dirname(utils.__file__) + "/activity_data.csv",
        os.path.dirname(utils.__file__)
        + "/fluorescence_df_for_chodera_lab.csv",
    )()

    list(iter(df.view(batch_size=32)))

    list(iter(df.view("all_graphs")))
