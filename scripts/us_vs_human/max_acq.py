import pinot
import torch
import numpy as np
import scipy



def run(args):

    ds = pinot.data.esol()

    ds = pinot.data.utils.batch(ds, len(ds))

    g_all, y_all = ds[0]

    net = pinot.Net(
        representation=pinot.representation.Sequential(
            layer=pinot.representation.dgl_legacy.gn(),
            config=[32, 'tanh', 32, 'tanh', 32, 'tanh']),
            output_regressor=pinot.regressors.ExactGaussianProcessRegressor)

    optimizer = torch.optim.Adam(net.parameters(), 1e-3)

    acquisition = getattr(
        pinot.active.acquisition.
        args.acquisition)

    bo = pinot.active.experiment.BayesOptExperiment(
        net=net,
        acquisition=acquisition,
        optimizer=optimizer,
        strategy=args.strategy,
        q=args.q,
        data=ds[0],
        slice_fn=pinot.active.experiment._slice_fn_tuple, # pinot.active.
        collate_fn=pinot.active.experiment._collate_fn_graph, # pinot.active.
        n_epochs=args.n_epochs,
        )

    xs = []
    num_rounds = args.num_rounds

    idx = 0
    bo.blind_pick(seed=None)
    bo.update_data()

    while idx < num_rounds:
        bo.train()
        bo.acquire()
        bo.update_data()


        x = pinot.active.acquisition.probability_of_improvement(
            pinot.metrics._independent(bo.net.condition(g_all)),
            y_best=bo.y_best).max().detach().squeeze().numpy()

        xs.append(x)

        idx += 1

    xs = np.array(xs)

    np.save(
        str(args.acquisition) + '_' + str(args.index) + '.npy',
        xs)

if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--acquisition', type=str, default='probability_of_improvement')
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--strategy', type=str, default='sequential')
    parser.add_argument('--num_rounds', type=str, default=10)
    parser.add_argument('--n_epochs', type=str, default=10)
