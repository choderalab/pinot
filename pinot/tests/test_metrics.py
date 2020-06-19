import pytest
import numpy as np
import numpy.testing as npt
import torch


def test_import():
    import pinot.metrics


def test_mse_and_rmse():
    import pinot

    # we only test here that:
    #  * the implementation is consistent with numpy
    #  * rmse is root of mse
    for _ in range(5):
        x = np.random.normal(size=(10,))
        y = np.random.normal(size=(10,))

        npt.assert_almost_equal(
            np.mean((x - y) ** 2),
            pinot.metrics._mse(torch.tensor(x), torch.tensor(y))
            .detach()
            .numpy(),
        )

        npt.assert_almost_equal(
            pinot.metrics._mse(torch.tensor(x), torch.tensor(y))
            .pow(0.5)
            .detach()
            .numpy(),
            pinot.metrics._rmse(torch.tensor(x), torch.tensor(y))
            .detach()
            .numpy(),
        )


def test_r2():
    import pinot

    x = np.random.normal(size=(10,))
    y = x

    npt.assert_almost_equal(
        pinot.metrics._r2(torch.tensor(x), torch.tensor(y)).detach().numpy(),
        1.0,
    )
