import pytest
import torch


def test_import():
    """ """
    import pinot
    import pinot.active
    import pinot.active.acquisition
    from pinot.active.acquisition import upper_confidence_bound
    from pinot.active.acquisition import probability_of_improvement
    from pinot.active.acquisition import expected_improvement_analytical
    from pinot.active.acquisition import expected_improvement_monte_carlo


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

    ei = expected_improvement_monte_carlo(normal, y_best=0.0, n_samples=100000)
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
