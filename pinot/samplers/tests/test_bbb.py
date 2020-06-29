import pytest
import pinot
import torch
import numpy as np
import numpy.testing as npt


def test_import():
    """ """
    import pinot.samplers.bbb


def test_linear_regression():
    """ """
    x = torch.distributions.normal.Normal(loc=0.0, scale=10.0).sample([100])

    y = 2 * x + 1.0

    k = torch.tensor(0.0)
    k.requires_grad = True

    b = torch.tensor(0.0)
    b.requires_grad = True

    opt = torch.optim.Adam([k, b], 1e-1)
    bbb = pinot.samplers.bbb.BBB(opt, 1e-5)

    for _ in range(3000):

        def l():
            """ """
            bbb.zero_grad()
            y_hat = k * x + b
            loss = (y - y_hat).pow(2).sum()
            loss.backward()
            return loss

        bbb.step(l)

    bbb.expectation_params()

    npt.assert_almost_equal(k.detach().numpy(), 2.0, decimal=1)

    npt.assert_almost_equal(b.detach().numpy(), 1.0, decimal=1)
