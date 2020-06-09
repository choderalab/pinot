import pytest
import torch

def test_import():
    import pinot
    import pinot.active.experiment

@pytest.fixture
def bo():
    import pinot

    def f(x):
        """ Example from
        https://pyro.ai/examples/bo.html
        """
        return (6 * x - 2)**2 * torch.sin(12 * x - 4)

    x = torch.linspace(0, 1)[:, None]
    y = f(x)

    net = pinot.Net(
        representation=torch.nn.Sequential(
            torch.nn.Linear(1, 50),
            torch.nn.Tanh()))

    return pinot.active.experiment.SingleTaskBayesianOptimizationExperiment(
        net=net,
        data=torch.stack([x, y], dim=1),
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        acquisition=pinot.active.acquisition.probability_of_improvement,
        n_epochs_training=10)

def test_init(bo):
    bo

def test_reset(bo):
    bo.reset_net()

def test_blinkd_pick(bo):
    assert isinstance(bo.blind_pick(), int)

def test_train(bo):
    bo.blind_pick()
    bo.train()

def test_run(bo):
    old = bo.run(limit=2)
