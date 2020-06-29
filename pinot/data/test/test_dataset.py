import pytest


def test_import():

    import pinot.data.datasets


@pytest.fixture
def esol():

    import pinot

    ds = pinot.data.esol()

    return ds


def test_esol(esol):

    esol
