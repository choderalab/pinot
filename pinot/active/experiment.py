# =============================================================================
# IMPORTS
# =============================================================================
import gc
import torch
from torch.utils.data import WeightedRandomSampler
import abc
import dgl
import pinot
from pinot.generative.utils import (
    batch_semi_supervised,
    prepare_semi_supervised_data,
)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    return torch.distributions.normal.Normal(
        distribution.mean.flatten(), distribution.variance.flatten().pow(0.5)
    )


def _slice_fn_tensor(x, idxs):
    return x[idxs]


def _collate_fn_tensor(x):
    return torch.stack(x)


def _collate_fn_graph(x):
    return dgl.batch(x)


def _slice_fn_graph(x, idxs):
    if x.batch_size > 1:
        x = dgl.unbatch(x)
    return dgl.batch([x[idx] for idx in idxs])


def _slice_fn_tuple(x, idxs):
    gs, ys = x
    graph_slices = _slice_fn_graph(gs, idxs)
    tensor_slices = _slice_fn_tensor(ys, idxs)
    return graph_slices, tensor_slices


# =============================================================================
# MODULE CLASSES
# =============================================================================
class ActiveLearningExperiment(torch.nn.Module, abc.ABC):
    """ Implements active learning experiment base class.
    """

    def __init__(self):
        super(ActiveLearningExperiment, self).__init__()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        raise NotImplementedError

    @abc.abstractmethod
    def acquire(self, *args, **kwargs):
        raise NotImplementedError


class BayesOptExperiment(ActiveLearningExperiment):
    """ Implements active learning experiment with single task target.
    """

    def __init__(
        self,
        net,
        data,
        acquisition,
        optimizer,
        n_epochs=100,
        strategy="sequential",
        q=1,
        num_samples=1000,
        early_stopping=True,
        weighted_acquire=False,
        workup=_independent,
        slice_fn=_slice_fn_tensor,
        collate_fn=_collate_fn_tensor,
        net_state_dict=None,
    ):

        super(BayesOptExperiment, self).__init__()

        # model
        self.net = net
        self.optimizer = optimizer
        self.n_epochs = n_epochs

        # data
        self.data = data
        self.old = []
        if isinstance(data, tuple):
            self.new = list(range(len(data[1])))
        else:
            self.new = list(range(len(data)))

        # acquisition
        self.acquisition = acquisition
        self.strategy = strategy

        # batch acquisition stuff
        self.q = q
        self.num_samples = num_samples
        self.weighted_acquire = weighted_acquire

        # early stopping
        self.early_stopping = early_stopping
        self.best_possible = torch.max(self.data[1])

        # bookkeeping
        self.workup = workup
        self.slice_fn = slice_fn
        self.collate_fn = collate_fn
        self.net_state_dict = net_state_dict

    def reset_net(self):
        """ Reset everything.
        """
        # TODO:
        # reset optimizer too
        (p.reset_parameters() for _, p in self.net.named_children())

        if self.net_state_dict is not None:
            self.net.load_state_dict(self.net_state_dict)

    def blind_pick(self, seed=2666):
        import random

        random.seed(seed)
        best = random.choice(self.new)
        self.old.append(self.new.pop(best))
        return best

    def train(self):
        """ Train the model with new data.
        """
        # reset
        self.reset_net()

        # set to train status
        self.net.train()

        # train the model
        self.net = pinot.app.experiment.Train(
            data=[self.old_data],
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            net=self.net,
            record_interval=999999,
        ).train()

    def acquire(self):
        """ Acquire new training data.
        """
        # set net to eval
        self.net.eval()

        # split input target
        gs, ys = self.new_data

        # get the predictive distribution
        # TODO:
        # write API for sampler
        distribution = self.net.condition(gs)

        if self.strategy == "batch":

            # batch acquisition
            indices, q_samples = self.acquisition(
                posterior=distribution,
                batch_size=gs.batch_size,
                y_best=self.y_best,
            )

            # argmax sample batch
            best = indices[:, torch.argmax(q_samples)].squeeze()

        else:
            # workup
            distribution = self.workup(distribution)

            # get score
            score = self.acquisition(distribution, y_best=self.y_best)

            if not self.weighted_acquire:
                # argmax
                best = torch.topk(score, self.q).indices
            else:
                # generate probability distribution
                weights = torch.exp(-score)
                weights = weights/weights.sum()
                best = WeightedRandomSampler(
                    weights=weights,
                    num_samples=self.q,
                    replacement=False)

        # pop from the back so you don't disrupt the order
        best = best.sort(descending=True).values
        # print(len(self.new), best)
        self.old.extend([self.new.pop(b) for b in best])

    def update_data(self):
        """ Update the internal data using old and new.
        """
        # grab new data
        self.new_data = self.slice_fn(self.data, self.new)

        # grab old data
        self.old_data = self.slice_fn(self.data, self.old)

        # set y_max
        gs, ys = self.old_data
        self.y_best = torch.max(ys)

    def run(self, num_rounds=999999, seed=None):
        """ Run the model.
        Parameters
        ----------
        rounds : int, default=99999
        """
        idx = 0
        self.blind_pick(seed=seed)
        self.update_data()

        while idx < num_rounds and len(self.new) > 0:
            self.train()
            self.acquire()
            self.update_data()

            if self.early_stopping and self.y_best == self.best_possible:
                break

            idx += 1

        return self.old


class SemiSupervisedBayesOptExperiment(BayesOptExperiment):
    """ Implements active learning experiment with single task target.
    """

    def __init__(self, *args, **kwargs):

        super(SemiSupervisedBayesOptExperiment, self).__init__(*args, **kwargs)

    def train(self):
        """ Train the model with new data.
        """
        # combine new (unlabeled!) and old (labeled!)
        # Flatten the labeled_data and remove labels to be ready
        semi_supervised_data = prepare_semi_supervised_data(
            self.flatten_data(self.new_data), self.flatten_data(self.old_data)
        )

        # NOTE that we have to use a special version of batch here
        # because torch.tensor doesn't take in `None`
        batched_semi_supervised_data = batch_semi_supervised(
            semi_supervised_data, batch_size=len(semi_supervised_data)
        )

        # reset
        self.reset_net()

        # set to train status
        self.net.train()

        # train the model
        self.net = pinot.app.experiment.Train(
            data=batched_semi_supervised_data,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            net=self.net,
            record_interval=999999,
        ).train()

    def flatten_data(self, data):
        gs, ys = data
        # if gs.batch_size > 1:
        gs = dgl.unbatch(gs)

        flattened_data = list(zip(gs, list(ys)))
        return flattened_data