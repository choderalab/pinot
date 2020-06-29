import pytest

def test_import():
    """ """
    import pinot.data.datasets


@pytest.fixture
def moonshot():
    """ """
    import pinot
    ds = pinot.data.moonshot_with_date
    return ds


def test_moonshot(moonshot):
    """

    Parameters
    ----------
    moonshot :
        

    Returns
    -------

    """
    print(moonshot)
