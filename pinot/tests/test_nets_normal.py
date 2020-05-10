import pytest
import numpy as np
import numpy.testing as npt


def test_nets_normal_shallow():
    """ In this test, there is no deep neural networks involved:
    only the parameters of the distribution: `mu` and
    `mu` are trained.


    """
    import torch
    import pinot

    # initialize parameter to be trained
    # $ \theta = \{ s
    theta = torch.distributions.normal.Normal(
        loc=10 * torch.ones(size=(1, 2)), scale=10 * torch.ones(size=(1, 2))
    ).sample()

    theta.requires_grad = True

    # initialize a simple model
    net = pinot.Net(
        representation=lambda x: None,
        parameterization=lambda x: theta,
        distribution_class=torch.distributions.normal.Normal,
    )

    # train for some epochs
    n_epochs = 100

    opt = torch.optim.LBFGS([theta], 0.1, line_search_fn="strong_wolfe")

    y = torch.distributions.normal.Normal(
        loc=torch.zeros(size=(1024,)), scale=torch.ones(size=(1024,))
    ).sample()

    for _ in range(n_epochs):
        # maximize the likelihood of samples
        # drawn from a true Gaussian
        def l():
            loss = net.loss(None, y).sum()

            opt.zero_grad()
            loss.backward()
            return loss

        opt.step(l)

    npt.assert_almost_equal(
        theta[0, 0].detach().numpy(), torch.mean(y).detach().numpy(), decimal=3
    )

    npt.assert_almost_equal(
        theta[0, 1].detach().numpy(),
        torch.log(torch.std(y)).detach().numpy(),
        decimal=3,
    )


def test_nets_normal_deep():
    """ In this test, we test the ability of the infrastructure to 
    parameterize a deep neural network to infer the parameters of a 
    known distribution.


    """
    import torch
    import pinot

    x = torch.ones(size=(32,))

    # initialize a simple model
    net = pinot.Net(
        representation=torch.nn.Sequential(torch.nn.Linear(32, 32), torch.nn.Tanh()),
        parameterization=torch.nn.Linear(32, 2),
        distribution_class=torch.distributions.normal.Normal,
    )

    # train for some epochs
    n_epochs = 100

    opt = torch.optim.LBFGS(net.parameters(), 0.1, line_search_fn="strong_wolfe")

    y = torch.distributions.normal.Normal(
        loc=torch.zeros(size=(1024,)), scale=torch.ones(size=(1024,))
    ).sample()

    for _ in range(n_epochs):
        # maximize the likelihood of samples
        # drawn from a true Gaussian
        def l():
            loss = net.loss(x, y).sum()

            opt.zero_grad()
            loss.backward()
            return loss

        opt.step(l)

    theta = net.forward(x)

    npt.assert_almost_equal(
        theta[0].detach().numpy(), torch.mean(y).detach().numpy(), decimal=3
    )

    npt.assert_almost_equal(
        theta[1].detach().numpy(), torch.log(torch.std(y)).detach().numpy(), decimal=3
    )
