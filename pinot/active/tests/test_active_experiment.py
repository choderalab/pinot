import pytest
import torch


def test_import():
    """ """
    import pinot
    import pinot.active.experiment


@pytest.fixture
def bo():
    """ """
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

    return pinot.active.experiment.BayesOptExperiment(
        net=net,
        data=torch.cat([x, y], dim=1),
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        acquisition=pinot.active.acquisition.probability_of_improvement,
        n_epochs=10,
    )


def test_init(bo):
    """

    Parameters
    ----------
    bo :
        

    Returns
    -------

    """
    bo


def test_reset(bo):
    """

    Parameters
    ----------
    bo :
        

    Returns
    -------

    """
    bo.reset_net()


def test_blinkd_pick(bo):
    """

    Parameters
    ----------
    bo :
        

    Returns
    -------

    """
    assert isinstance(bo.blind_pick(), int)
