import pytest
import numpy as np
import numpy.testing as npt
import torch

@pytest.mark.parametrize(
    "y_fn", [
        lambda x: torch.flatten(-2.0 * x + 1.0), # no noise
        lambda x: torch.flatten(
            torch.distributions.normal.Normal(
                loc=-2.0 * x + 1.0,
                scale=1e-3).sample())])
def test_linear_regression_fixed_sigma(y_fn):
    """ We test the model is able to recover parameters
    through linear regression.


    """
    import torch
    import pinot

    x = torch.distributions.normal.Normal(
            loc=torch.zeros(size=(256, 1)),
            scale=10 * torch.ones(size=(256, 1))).sample()

    y = y_fn(x)

    # initialize a simple model
    net = pinot.Net(
            representation=lambda x: x,
            parameterization=torch.nn.Linear(1, 2),
            distribution_class=torch.distributions.normal.Normal,
            param_transform=lambda x, y: (x, 1.0))

    # train for some epochs
    n_epochs = 1000

    opt = torch.optim.Adam(net.parameters(), 1e-1)

    for _ in range(n_epochs):
        # maximize the likelihood of samples
        # drawn from a true Gaussian
        def l():
            loss = net.loss(x, y).mean()

            opt.zero_grad()
            loss.backward()
            return loss
        
        opt.step(l)
   
    theta = net.forward(x)

    npt.assert_almost_equal(
        net.parameterization.weight[0].detach().numpy(),
        -2.0,
        decimal=3)

    npt.assert_almost_equal(
        net.parameterization.bias[0].detach().numpy(),
        1.0,
        decimal=3)

