import pytest
import numpy as np
import numpy.testing as npt

def test_linear_regression_fixed_sigma():
    """ We test the model is able to recover parameters
    through linear regression.


    """
    import torch
    import pinot

    x = torch.distributions.normal.Normal(
            loc=torch.zeros(size=(256, 1)),
            scale=10 * torch.ones(size=(256, 1))).sample()

    y = -2.0 * x + 1.0

    # initialize a simple model
    net = pinot.Net(
            representation=lambda x: x,
            parameterization=torch.nn.Linear(1, 2),
            distribution_class=torch.distributions.normal.Normal,
            param_transform=lambda x, y: (x, 1.0))

    # train for some epochs
    n_epochs = 100

    opt = torch.optim.LBFGS(
            net.parameters(),
            0.1,
            line_search_fn='strong_wolfe')



    for _ in range(n_epochs):
        # maximize the likelihood of samples
        # drawn from a true Gaussian
        def l():
            loss = net.loss(x, y).sum()

            opt.zero_grad()
            loss.backward()
            print(loss, flush=True)
            return loss
        
        opt.step(l)
   
    theta = net.forward(x)

    print(list(net.parameters()))

    npt.assert_almost_equal(
        net.parameterization.weight[0].detach().numpy(),
        -2.0,
        decimal=3)

    npt.assert_almost_equal(
        net.parameterization.bias[0].detach().numpy(),
        1.0,
        decimal=3)

