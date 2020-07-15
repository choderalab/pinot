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


def test_mix_default():
    import pinot
    from pinot.data import utils
    import os

    df = pinot.data.moonshot_mixed()

    view = df.view("all_available_pairs", batch_size=32)

    for g, y, c in view:
        g
        y
        c


def test_rebatch():
    import pinot
    from pinot.data import utils
    import os

    df = pinot.data.moonshot_mixed()

    view = df.view("fixed_size_batch", batch_size=32, drop_last=True)

    df.number_of_measurements

    for g, c, y in view:
        assert g.batch_size == 32


def test_filter():
    import pinot
    from pinot.data import utils
    import os
    ds = pinot.data.moonshot_mixed()

    view = ds.view('fixed_size_batch_filter_20')

    for g, y, c in view:
        assert float(c) == float(20)
