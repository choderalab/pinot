import pytest
import torch


def test_import():
    """ """
    import pinot
    import pinot.active
    import pinot.active.acquisition

    # sequential methods
    from pinot.active.acquisition import upper_confidence_bound
    from pinot.active.acquisition import probability_of_improvement
    from pinot.active.acquisition import expected_improvement_analytical
    from pinot.active.acquisition import expected_improvement_monte_carlo

    # batch methods
    from pinot.active.acquisition import thompson_sampling
    from pinot.active.acquisition import random
    from pinot.active.acquisition import temporal
    from pinot.active.acquisition import uncertainty


@pytest.fixture
def normal():
    """ """
    return torch.distributions.normal.Normal(0.0, 1.0)

# BATCH
@pytest.fixture
def setup():

    import pinot

    def f(x):
        """Example from
        https://pyro.ai/examples/bo.html

        Parameters
        ----------
        x :
            

        Returns
        -------

        """
        return (6 * x - 2) ** 2 * torch.sin(12 * x - 4)

    x = torch.linspace(0, 1)[:, None]
    y = f(x)

    net = pinot.Net(
        representation=torch.nn.Sequential(
            torch.nn.Linear(1, 50), torch.nn.Tanh()
        )
    )

    unseen_data = x

    return net, unseen_data

def test_thompson_sampling(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import thompson_sampling
    thompson_sampling(*setup, q=5, y_best=0.0)

def test_batch_random(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import random
    random(*setup, q=5, y_best=0.0)

def test_batch_temporal(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import temporal
    ac = temporal(*setup, q=5, y_best=0.0)
    assert len(ac) == 5


def test_pi(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import probability_of_improvement

    pi = probability_of_improvement(*setup, q=5, y_best=0.0)
    assert len(pi) == 5


def test_ei_analytical_normal(setup, normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import expected_improvement_analytical

    ei = expected_improvement_analytical(*setup, q=5, y_best=0.0)
    assert len(ei) == 5


def test_ei_monte_carlo(setup, normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import expected_improvement_monte_carlo

    ei = expected_improvement_monte_carlo(
        *setup, q=5, y_best=0.0, n_samples=100000
    )
    assert len(ei) == 5


def test_ucb(setup, normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import upper_confidence_bound

    ucb = upper_confidence_bound(*setup, q=5, y_best=0.0, kappa=0.95)
    assert len(ucb) == 5