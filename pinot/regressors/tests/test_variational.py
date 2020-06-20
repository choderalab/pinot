import pytest
import torch
import pinot

def test_import():
    from pinot.regressors import VariationalGaussianProcessRegressor

def test_init():
    from pinot.regressors import VariationalGaussianProcessRegressor
    ds_tr, ds_te, num_data = get_data()

    layer = pinot.representation.dgl_legacy.GN
    net_representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )

    net = pinot.Net(
        net_representation,
        VariationalGaussianProcessRegressor,
        num_data=num_data
    )        

def test_train_and_test():
    from pinot.regressors import VariationalGaussianProcessRegressor
    ds_tr, ds_te, num_data = get_data()

    layer = pinot.representation.dgl_legacy.GN
    representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )

    net = pinot.Net(
        representation,
        VariationalGaussianProcessRegressor,
        num_data=num_data
    )

    lr = 1e-4
    optimizer = torch.optim.Adam([
        {'params': net.representation.parameters(), 'weight_decay': lr},
        {'params': net.output_regressor.output_regressor.hyperparameters(), 'lr': lr * 0.01},
        {'params': net.output_regressor.output_regressor.variational_parameters()},
        {'params': net.output_regressor.likelihood.parameters()}
    ])

    train_and_test = pinot.TrainAndTest(
        net=net,
        optimizer=optimizer,
        n_epochs=1,
        data_tr=ds_tr,
        data_te=ds_te,
    )

    print(train_and_test)


def test_train_and_test_cuda():
    from pinot.regressors import VariationalGaussianProcessRegressor
    ds_tr, ds_te, num_data = get_data(cuda=True)

    layer = pinot.representation.dgl_legacy.GN
    representation = pinot.representation.Sequential(
        layer, [32, "tanh", 32, "tanh", 32, "tanh"]
    )

    net = pinot.Net(
        representation,
        VariationalGaussianProcessRegressor,
        num_data=num_data
    ).to(torch.device('cuda:0'))

    lr = 1e-4
    optimizer = torch.optim.Adam([
        {'params': net.representation.parameters(), 'weight_decay': lr},
        {'params': net.output_regressor.output_regressor.hyperparameters(), 'lr': lr * 0.01},
        {'params': net.output_regressor.output_regressor.variational_parameters()},
        {'params': net.output_regressor.likelihood.parameters()}
    ])

    train_and_test = pinot.TrainAndTest(
        net=net,
        optimizer=optimizer,
        n_epochs=1,
        data_tr=ds_tr,
        data_te=ds_te,
    )

    print(train_and_test)



def get_data(cuda=False):
    # get data
    ds = pinot.data.esol()
    num_data = len(ds)
    if cuda:
        ds_new = []
        for d in ds:
            d = tuple([i.to(torch.device('cuda:0')) for i in d])
            ds_new.append(d)
        ds = ds_new
    ds = pinot.data.utils.batch(ds, 64)
    # see if normalizing does anything
    ds_tr, ds_te = pinot.data.utils.split(ds, [4, 1])
    return ds_tr, ds_te, num_data