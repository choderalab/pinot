import pinot
import numpy as np
import torch

def probability_of_improvement(distribution, y_best=0.0):

    return 1.0 - distribution.cdf(y_best)

def expected_improvement(distribution, y_best=0.0):
    return distribution.mean - y_best

def upper_confidence_bound(distribution, y_best=0.0, kappa=0.5):
    from pinot.inference.utils import confidence_interval
    _, high = confidence_interval(distribution, kappa)
    return high

def _independent(distribution):
    return torch.distributions.normal.Normal(
        distribution.mean.flatten(),
        distribution.variance.pow(0.5).flatten())

def _slice_fn_tensor(data, idxs):
    # data.shape = (N, 2)
    # idx is a list
    data = data[idxs]
    assert data.dim() == 2
    assert data.shape[-1] == 2
    return data[:, 0][:, None], data[:, 1][:, None]

def _slice_fn_graph(data, idxs):
    # data is a list
    # idx is a list
    data = [data[idx] for idx in idxs]
    gs, ys = list(zip(*data))
    import dgl
    gs = dgl.batch(gs)
    ys = torch.stack(ys, dim=0)
    return gs, ys

class MultiTaskBayesianOptimizationExperiment(torch.nn.Module):
    """ Multitask BO.
    """
    def __init__(
        self,
        nets,
        datasets,
        acquisition,
        optimizer,
        limit=100,
        n_epochs_training=100,
        workup=_independent,
        slice_fn=_slice_fn_graph,
        net_state_dicts=None):

        self.nets = nets
        self.optimizer = optimizer
        self.n_epochs_training = n_epochs_training

        self.acquisition = acquisition

        self.worup = workup
        self.slice_fn = slice_fn


        self.limit = limit

        self.olds = []
        self.news = []

        self.datasets = datasets

        self.net_state_dicts = net_state_dicts


        self.workup = workup

        self.n_tasks = len(self.nets)

        for idx in range(self.n_tasks):
            self.olds.append([])
            self.news.append(list(range(len(datasets[idx]))))

        self.y_bests = [0 for x in range(self.n_tasks)]

    def reset_net(self):
        # TODO:
        # reset optimizer too

        for net in self.nets:
            for module in net.modules():
                if isinstance(module, torch.nn.Linear):
                    module.reset_parameters()

        if self.net_state_dicts is not None:
            for idx, net in enumerate(self.nets):
                net.load_state_dict(self.net_state_dicts[idx])

    def blind_pick(self):
        import random

        for idx in range(self.n_tasks):
            best = random.choice(self.news[idx])
            self.olds[idx].append(self.news[idx].pop(best))

    def train(self):
        """ Train the model with new data.
        """
        # reset
        self.reset_net()

        # set to train status
        [net.train() for net in self.nets]

        # grab old data
        # (N, 2) for tensor
        # N list of 2-tuple for lists
        old_datas = [self.slice_fn(self.datasets[idx], self.olds[idx]) for idx in range(self.n_tasks)]

        for _ in range(self.n_epochs_training):
            self.optimizer.zero_grad()
            loss = 0.0
            for idx in range(self.n_tasks):
                g, y = old_datas[0]
                net = self.nets[idx]
                _loss = net.loss(g, y).mean()
                loss += _loss
            loss.backward()
            self.optimizer.step()

        for idx in range(self.n_tasks):
            gs, ys = old_datas[idx]
            self.y_bests[idx] = torch.max(ys)


    def acquire(self):
        for idx, net in enumerate(self.nets):
            if len(self.news[idx]) > 0:
                gs, _ = self.slice_fn(self.datasets[idx], self.news[idx])
                distribution = net.condition(gs)
                distribution = self.workup(distribution)
                score = self.acquisition(distribution, y_best=self.y_bests[idx])


                best = torch.argmax(score)
                self.olds[idx].append(self.news[idx].pop(best))


    def run(self):
        self.blind_pick()

        step = 0
        while any(len(self.news[idx]) > 0 for idx in range(self.n_tasks)) and step < self.limit:
            bo.train()
            bo.acquire()
            step += 1





def run(path):
    ds = pinot.data.moonshot_meta()

    def get_separate_dataset(ds):
        # n_tasks = ds[0][1].shape[-1]
        n_tasks = 6
        datasets = [[] for _ in range(n_tasks)]

        for g, y in ds:
            for idx in range(n_tasks):
                if np.isnan(y[idx].numpy()) == False:
                    datasets[idx].append((g, y[idx][None]))

        return datasets


    datasets = get_separate_dataset(ds)

    net = pinot.representation.Sequential(
        pinot.representation.dgl_legacy.gn(),
        [32, 'tanh', 32, 'tanh', 32, 'tanh'])

    nets = []
    for idx in range(len(datasets)):

        model = pinot.inference.gp.gpr.exact_gpr.ExactGPR(
            kernel=pinot.inference.gp.kernels.deep_kernel.DeepKernel(
                base_kernel=pinot.inference.gp.kernels.rbf.RBF(torch.ones(32)),
                representation=net))

        nets.append(model)

    params = []
    for net in nets:
        params += list(net.parameters())

    bo = MultiTaskBayesianOptimizationExperiment(
        nets=nets,
        datasets=datasets,
        acquisition=probability_of_improvement,
        optimizer=torch.optim.Adam(params, 1e-5))

    bo.run()

    regrets = []
    for idx in range(bo.n_tasks):
        actual_best = max(bo.datasets[idx][idx_][1].squeeze() for idx_ in range(len(bo.datasets[idx]))).detach().numpy()

        regret = []

        for step in range(1, len(bo.olds[idx])):

            idxs = bo.olds[idx][:step]

            _, ys = bo.slice_fn(bo.datasets[idx], idxs)

            y_best_now = torch.max(ys.flatten()).detach().numpy()

            regret.append(actual_best - y_best_now)

        regrets.append(regret)

    import pickle
    with open(path + '/regret.dat', 'wb') as f_handle:
        pickle.dump(regrets, f_handle)


if __name__ == '__main__':
    import sys
    path = sys.argv[1]
    run(path)
