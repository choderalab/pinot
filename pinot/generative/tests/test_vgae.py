import pytest
import torch
import numpy as np
import numpy.testing as npt

def test_import():
    import pinot
    vgae = pinot.generative.VGAE()

@pytest.fixture
def vgae():
    import pinot
    return pinot.generative.VGAE()

def test_parameterization(vgae):
    x = torch.distributions.normal.Normal(
            loc=torch.zeros((32, 16)),
            scale=torch.ones((32, 16))).sample()

    mu, log_sigma = vgae.parametrization(x)

    assert mu.shape == torch.Size([32, 16])
    assert log_sigma.shape == torch.Size([32, 16])

    # kill stochasicity to center the latent space
    # distribution

    vgae.linear = lambda x: torch.stack(
            [
                torch.zeros_like(x),
                torch.ones_like(x)
            ],
            dim=-1)

    mu, log_sigma = vgae.parametrization(x)

    assert mu == torch.zeros_like(x)
    assert log_sigma == torch.ones_like(x)

def test_inference(vgae):
    
    q_z_i = vgae.inference(
            torch.zeros((32, 16)),
            torch.ones((32, 16)))

    assert isinstance(q_z_i, torch.distributions.normal.Normal)
    assert torch.equal(q_z_i.loc, torch.zeros((32, 16)))
    assert torch.equal(q_z_i.scale, torch.exp(torch.ones((32, 16))))

def test_log_p_a_given_z(vgae):
    # define a graph with only self connection
    a = torch.eye(5)

    # define z to be all ones
    z = torch.zeros((5, 32))

    log_p_a_given_z = vgae.log_p_a_given_z(a, z)

    # this should be exactly (5 - 20) = -15
    npt.assert_almost_equal(
        log_p_a_given_z,
        25 * np.log(0.5),
        decimal=2)
    

