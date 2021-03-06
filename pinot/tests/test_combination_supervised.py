import pytest
import pinot
import torch


@pytest.fixture
def representation():
    """ """
    layer = pinot.representation.dgl_legacy.gn()
    representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )

    return representation


@pytest.fixture
def ds():
    """ """
    ds = pinot.data.esol()[:8]
    ds = pinot.data.utils.batch(ds, 4)
    return ds


@pytest.fixture
def neural_network_regressor():
    """ """
    return pinot.regressors.NeuralNetworkRegressor


@pytest.fixture
def variational_gaussian_process_regressor():
    """ """
    return pinot.regressors.VariationalGaussianProcessRegressor


@pytest.fixture
def exact_gaussian_process_regressor():
    """ """
    return pinot.regressors.ExactGaussianProcessRegressor


@pytest.fixture
def vanilla_net():
    """ """
    return pinot.Net


@pytest.fixture
def semisupervised_net():
    """ """
    return pinot.generative.semi_supervised_net.SemiSupervisedNet


def test_import(
    neural_network_regressor,
    variational_gaussian_process_regressor,
    exact_gaussian_process_regressor,
    vanilla_net,
    semisupervised_net,
):
    """

    Parameters
    ----------
    neural_network_regressor :

    variational_gaussian_process_regressor :

    exact_gaussian_process_regressor :

    vanilla_net :

    semisupervised_net :


    Returns
    -------

    """
    neural_network_regressor
    variational_gaussian_process_regressor
    exact_gaussian_process_regressor
    vanilla_net
    semisupervised_net


@pytest.mark.parametrize(
    "net", [pinot.Net, pinot.generative.semi_supervised_net.SemiSupervisedNet]
)
@pytest.mark.parametrize(
    "regressor",
    [
        pinot.regressors.ExactGaussianProcessRegressor,
        pinot.regressors.VariationalGaussianProcessRegressor,
        pinot.regressors.NeuralNetworkRegressor,
    ],
)
def test_init(net, regressor, representation):
    """

    Parameters
    ----------
    net :

    regressor :

    representation :


    Returns
    -------

    """
    net(output_regressor_class=regressor, representation=representation)


@pytest.mark.parametrize("net", [pinot.Net,])
@pytest.mark.parametrize(
    "regressor",
    [
        pinot.regressors.ExactGaussianProcessRegressor,
        pinot.regressors.VariationalGaussianProcessRegressor,
        pinot.regressors.NeuralNetworkRegressor,
    ],
)
def test_forward(net, regressor, representation, ds):
    """

    Parameters
    ----------
    net :

    regressor :

    representation :

    ds :


    Returns
    -------

    """
    net = net(output_regressor_class=regressor, representation=representation)

    g, y = ds[0]
    loss = net.loss(g, y)

    distribution = net.condition(g)

    from pinot.metrics import _independent

    distribution = _independent(distribution)
    assert distribution.batch_shape == torch.Size([4])
    assert distribution.event_shape == torch.Size([])
