import pytest

def test_import():
    import pinot.data.datasets


@pytest.fixture
def esol():
    import pinot
    ds = pinot.data.datasets.Dataset(
            ).from_csv('../esol.csv')

    return ds

def test_esol(esol):
    esol
