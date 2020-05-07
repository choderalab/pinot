import pytest
import torch

def test_import():
    import pinot.app.experiment

def test_train():
    import pinot

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer,
        [32, 'tanh', 32, 'tanh', 32, 'tanh'])
    net_regression = pinot.regression.Linear(32, 2)
    net = pinot.Net(
        net_representation,
        net_regression)

    train = pinot.Train(
        net=net,
        data=pinot.data.esol()[:10],
        n_epochs=1,
        optimizer=torch.optim.Adam(net.parameters()))

    train.train()

def test_test():
    import pinot
    import copy

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer,
        [32, 'tanh', 32, 'tanh', 32, 'tanh'])
    net_regression = pinot.regression.Linear(32, 2)
    net = pinot.Net(
        net_representation,
        net_regression)

    train = pinot.Train(
        net=net,
        data=pinot.data.utils.batch(
            pinot.data.esol()[:10],
            5),
        n_epochs=1,
        optimizer=torch.optim.Adam(net.parameters()))

    train.train()

    test = pinot.Test(
        net=net,
        data=pinot.data.utils.batch(
            pinot.data.esol()[:10],
            5),
        metrics=[pinot.mse, pinot.rmse, pinot.r2],
        states=train.states)

    test.test()

def test_train_and_test():
    import pinot
    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer,
        [32, 'tanh', 32, 'tanh', 32, 'tanh'])
    net_regression = pinot.regression.Linear(32, 2)
    net = pinot.Net(
        net_representation,
        net_regression)

    train_and_test = pinot.TrainAndTest(
        net=net,
        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
        n_epochs=1,
        data_tr=pinot.data.utils.batch(pinot.data.esol()[:10], 5),
        data_te=pinot.data.utils.batch(pinot.data.esol()[:10], 5))


