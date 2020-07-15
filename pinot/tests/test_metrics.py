import pytest
import numpy as np
import numpy.testing as npt
import torch


def test_import():
    """ """
    import pinot.metrics

@pytest.fixture
def net():
    class Net(torch.nn.Module):
        def __init__(self):
            super(Net, self).__init__()

        def condition(self, *args, **kwargs):
            return torch.distributions.normal.Normal(10, 1)


    return Net()


def test_r2(net):
    import pinot
    g = None
    y = torch.distributions.normal.Normal(
            torch.zeros(10, 1),
            torch.ones(10, 1),
            ).sample()
    r2 = pinot.metrics.r2(net, g, y)


def test_rmse(net):
    import pinot
    g = None
    y = torch.distributions.normal.Normal(
            torch.zeros(10, 1),
            torch.ones(10, 1),
            ).sample()
    rmse = pinot.metrics.rmse(net, g, y)
