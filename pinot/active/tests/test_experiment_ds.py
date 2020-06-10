import pytest
import torch

def test_import():
    import pinot
    import pinot.active.experiment

@pytest.fixture
def bo():

    args = {
    'layer': 'GraphConv',
    'noise_model': 'normal-heteroschedastic',
    'optimizer': 'adam',
    'config': [32, 'tanh', 32, 'tanh', 32, 'tanh'],
    'out': 'result',
    'data': 'esol',
    'batch_size': 32,
    'opt': 'Adam',
    'lr': 1e-5,
    'partition': '4:1',
    'n_epochs': 40
    }

    import pinot

    # data
    ds = pinot.data.esol()
    ds = pinot.data.utils.batch(ds, len(ds), seed=None)

    # net
    layer = pinot.representation.dgl_legacy.gn(model_name=args['layer'])
    net_representation = pinot.representation.Sequential(
        layer=layer,
        config=args['config'])
    kernel = pinot.inference.gp.kernels.deep_kernel.DeepKernel(
            representation=net_representation,
            base_kernel=pinot.inference.gp.kernels.rbf.RBF())
    net = pinot.inference.gp.gpr.exact_gpr.ExactGPR(
            kernel)

    return pinot.active.experiment.SingleTaskBayesianOptimizationExperiment(
        net=net,
        data=ds[0],
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        acquisition=pinot.active.acquisition.probability_of_improvement,
        n_epochs_training=10,
        slice_fn = pinot.active.experiment._slice_fn_tuple,
        collate_fn = pinot.active.experiment._collate_fn_graph
        )

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