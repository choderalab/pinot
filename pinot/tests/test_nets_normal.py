import pytest
import numpy as np
import numpy.testing as npt
import torch
import pinot


def test_shallow_normal():
    net_representation = torch.nn.Linear(8, 2)
    net = pinot.Net(net_representation)

    x = torch.zeros(1024, 8)

    opt = torch.optim.LBFGS(net.parameters(), line_search_fn="strong_wolfe")

    # set target as a sample drawn from normal distrtibution
    y = torch.distributions.normal.Normal(torch.tensor(0.0), torch.tensor(1.0)).sample(
        [1024, 1]
    )

    for _ in range(10):

        def l():
            opt.zero_grad()
            loss = net.loss(x, y).sum()
            loss.backward()
            return loss

        opt.step(l)

    npt.assert_almost_equal(net.condition(x).loc.detach().numpy(), 0.0, decimal=1)

    npt.assert_almost_equal(net.condition(x).scale.detach().numpy(), 1.0, decimal=1)
