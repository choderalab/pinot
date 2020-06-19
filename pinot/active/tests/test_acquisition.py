import pytest
import torch


def test_import():
    import pinot
    import pinot.active
    import pinot.active.acquisition
    from pinot.active.acquisition import probability_of_improvement
    from pinot.active.acquisition import expected_improvement
    from pinot.active.acquisition import upper_confidence_bound


@pytest.fixture
def normal():
    return torch.distributions.normal.Normal(0.0, 1.0)


def test_pi(normal):
    from pinot.active.acquisition import probability_of_improvement

    pi = probability_of_improvement(normal)
    assert pi == 0.5


def test_ei(normal):
    from pinot.active.acquisition import expected_improvement

    ei = expected_improvement(normal)
    assert ei == 0.0


def test_ucb(normal):
    from pinot.active.acquisition import upper_confidence_bound

    ucb = upper_confidence_bound(normal, kappa=0.95)
    assert ucb == normal.icdf(torch.tensor(0.975))
