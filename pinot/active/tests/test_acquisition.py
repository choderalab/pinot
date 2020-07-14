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
    from pinot.active.acquisition import exponential_weighted_ei_analytical
    from pinot.active.acquisition import exponential_weighted_ei_monte_carlo
    from pinot.active.acquisition import exponential_weighted_pi
    from pinot.active.acquisition import exponential_weighted_ucb
    from pinot.active.acquisition import greedy_ucb
    from pinot.active.acquisition import greedy_ei_analytical
    from pinot.active.acquisition import greedy_ei_monte_carlo
    from pinot.active.acquisition import greedy_pi
    from pinot.active.acquisition import batch_random
    from pinot.active.acquisition import batch_temporal


@pytest.fixture
def normal():
    """ """
    return torch.distributions.normal.Normal(0.0, 1.0)


def test_pi(normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import probability_of_improvement

    pi = probability_of_improvement(normal)
    assert pi == 0.5


def test_ei_analytical_normal(normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import expected_improvement_analytical

    ei = expected_improvement_analytical(normal, y_best=0.0)
    assert torch.exp(normal.log_prob(0.0)) == ei


def test_ei_monte_carlo(normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import expected_improvement_monte_carlo

    ei = expected_improvement_monte_carlo(
        normal, y_best=0.0, n_samples=100000
    )
    assert (torch.exp(normal.log_prob(0.0)) - ei) < 1e-2


def test_ucb(normal):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import upper_confidence_bound

    ucb = upper_confidence_bound(normal, kappa=0.95)
    assert ucb == normal.icdf(torch.tensor(0.975))


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

    unseen_data = tuple([x, y])

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


def test_exponential_weighted_ei_analytical(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import exponential_weighted_ei_analytical
    exponential_weighted_ei_analytical(*setup, q=5, y_best=0.0)

def test_exponential_weighted_ei_monte_carlo(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import exponential_weighted_ei_monte_carlo
    exponential_weighted_ei_monte_carlo(*setup, q=5, y_best=0.0)

def test_exponential_weighted_pi(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import exponential_weighted_pi
    exponential_weighted_pi(*setup, q=5, y_best=0.0)

def test_exponential_weighted_ucb(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import exponential_weighted_ucb
    exponential_weighted_ucb(*setup, q=5, y_best=0.0)

def test_greedy_ucb(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import greedy_ucb
    greedy_ucb(*setup, q=5, y_best=0.0)

def test_greedy_ei_analytical(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import greedy_ei_analytical
    greedy_ei_analytical(*setup, q=5, y_best=0.0)

def test_greedy_ei_monte_carlo(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import greedy_ei_monte_carlo
    greedy_ei_monte_carlo(*setup, q=5, y_best=0.0)


def test_greedy_pi(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import greedy_pi
    greedy_pi(*setup, q=5, y_best=0.0)

def test_batch_random(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import batch_random
    batch_random(*setup, q=5, y_best=0.0)

def test_batch_temporal(setup):
    """

    Parameters
    ----------
    normal :
        

    Returns
    -------

    """
    from pinot.active.acquisition import batch_temporal
    batch_temporal(*setup, q=5, y_best=0.0)