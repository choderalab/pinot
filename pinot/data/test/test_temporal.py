import pytest

def test_import():
    import pinot.data.datasets


@pytest.fixture
def moonshot():
    import pinot
    ds = pinot.data.datasets.TemporalDataset(
            ).from_csv(
                '../moonshot_with_date.csv',
                smiles_col=1,
                y_cols=[5, 6, 7, 8, 9, 10],
                time_col=-3)

    return ds


def test_moonshot(moonshot):
    print(moonshot)
