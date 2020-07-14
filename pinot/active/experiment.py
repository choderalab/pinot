# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import dgl
import pinot
from pinot.metrics import _independent
from torch.utils.data import WeightedRandomSampler

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _slice_fn_tensor(x, idxs):
    """ Slice function for tensors.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(n_data, )`
        Input tensor.

    idxs : `List` of `int`
        Indices to be taken.

    Returns
    -------
    x : `torch.Tensor`, `shape=(n_data_chosen, )`
        Output tensor.

    """
    return x[idxs]

def _slice_fn_tensor_pair(x, idxs):
    """ Slice function for tensors.

    Parameters
    ----------
    x : `torch.Tensor`, `shape=(n_data, )`
        Input tensor.

    idxs : `List` of `int`
        Indices to be taken.

    Returns
    -------
    x : `torch.Tensor`, `shape=(n_data_chosen, )`
        Output tensor.

    """
    return x[0][idxs], x[1][idxs]

def _collate_fn_tensor(x):
    """ Collate function for tensors.

    Parameters
    ----------
    x : `List` of `torch.Tensor`
        Tensors to be stacked.


    Returns
    -------
    x : `torch.Tensor`
        Output tensor.

    """
    return torch.stack(x)


def _collate_fn_graph(x):
    """ Collate function for graphs.

    Parameters
    ----------
    x : `List` of `dgl.DGLGraph`
        Input list of graphs to be batched.


    Returns
    -------
    x : `dgl.DGLGraph`

    """
    return dgl.batch(x)


def _slice_fn_graph(x, idxs):
    """ Slice function for graphs.

    Parameters
    ----------
    x : `dgl.DGLGraph`
        Batched graph.

    idxs : `List` of `int`
        Indices of the chosen graphs.


    Returns
    -------
    x : `dgl.DGLGraph`
        Sliced graph.

    """
    if x.batch_size > 1:
        x = dgl.unbatch(x)
    else:
        raise RuntimeError("Can only slice if there is more than one.")

    return dgl.batch([x[idx] for idx in idxs])


def _slice_fn_tuple(x, idxs):
    """ Slice function for tuples.

    Parameters
    ----------
    x : `List` of `tuple`
        Input data pairs.

    idxs : `List` of `int`
        Indices of chosen data points.

    Returns
    -------
    `graph_slices` : `dgl.DGLGraph`
        Sliced and batched graph.

    `tensor_clies` : `torch.Tensor`
        Sliced and stacked tensor.

    """
    gs, ys = x
    graph_slices = _slice_fn_graph(gs, idxs)
    tensor_slices = _slice_fn_tensor(ys, idxs)
    return graph_slices, tensor_slices


# =============================================================================
# MODULE CLASSES
# =============================================================================
class ActiveLearningExperiment(torch.nn.Module, abc.ABC):
    """Implements active learning experiment base class."""

    def __init__(self):
        super(ActiveLearningExperiment, self).__init__()

    @abc.abstractmethod
    def train(self, *args, **kwargs):
        """ Train the model. """
        raise NotImplementedError

    @abc.abstractmethod
    def acquire(self, *args, **kwargs):
        """ Acquire new data points from pool or space. """
        raise NotImplementedError


class BayesOptExperiment(ActiveLearningExperiment):
    """Implements active learning experiment with single task target.

    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    acquisition : `callable`
        Acquisition function that takes the graphs of candidates and
        provides scores.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    n_epochs : `int`
        Number of epochs.

    strategy : `str`, either `sequential` or `batch`
        Acquisition strategy.

    q : `int`
        Number of acquired candidates in each round.

    early_stopping : `bool`
        Whether the search ends when the best within the candidate pool
        is already acquired.

    workup : `callable` (default `_independent`)
        Post-processing predictive distribution.

    slice_fn : `callable` (default `_slice_fn_tensor`)
        Function used to slice data.

    collate_fn : `callable` (default `_collate_fn_tensor`)
        Function used to stack or batch input.


    Methods
    -------
    reset_net : Reset the states of `net`.

    blind_pick : Pick random candidates to start acquiring.

    train : Train the model for some epochs in one round.

    acquire : Acquire candidates from pool.

    run : Conduct rounds of acquisition and train.

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
        weighted_acquire=False,
        early_stopping=True,
        workup=_independent,
        slice_fn=_slice_fn_tensor,
        collate_fn=_collate_fn_tensor,
        net_state_dict=None,
        train_class=pinot.app.experiment.Train,
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
            # If data is simply (batched_graph, all_measurements)
            self.new = list(range(len(data[1])))
            self.g_all = data[0]
        else:
            # If data is of the form list[(graph, measurement)]
            self.new = list(range(len(data)))
            self.g_all = dgl.batch([g for (g, y) in data]) 

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
        self.train_class = train_class

    def reset_net(self):
        """Reset everything."""
        # TODO:
        # reset optimizer too
        (p.reset_parameters() for _, p in self.net.named_children())

        if self.net_state_dict is not None:
            self.net.load_state_dict(self.net_state_dict)

    def blind_pick(self, seed=2666):
        """ Randomly pick index from the candidate pool.

        Parameters
        ----------
        seed : `int`
             (Default value = 2666)
             Random seed.

        Returns
        -------
        best : `int`
            The chosen candidate to start.

        Note
        ----
        Random seed set to `2666`, the title of the single greatest novel in
        human literary history by Roberto Bolano.
        This needs to be set to `None`
        if parallel experiments were to be performed.

        """
        import random

        random.seed(seed)
        best = random.choice(self.new)
        self.old.append(self.new.pop(best))
        return best

    def train(self):
        """Train the model with new data."""
        # reset
        self.reset_net()

        # set to train status
        self.net.train()

        # train the model
        self.net = self.train_class(
            data=[self.old_data],
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            net=self.net,
            record_interval=999999,
        ).train()

    def acquire(self):
        """Acquire new training data."""
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

            # Compute the max probability of improvement needed for later plots
            # Note that this probability of improvement is computed for all graphs
            prob_improvement = pinot.active.acquisition.probability_of_improvement(
                    pinot.metrics._independent(self.net.condition(self.g_all)),
                                y_best=self.y_best)
            print("Max probability of improvement:")
            print(prob_improvement.max().detach())
            if not self.weighted_acquire:
                # argmax
                best = torch.topk(score, min(self.q, len(score))).indices
            else:
                # generate probability distribution
                weights = torch.exp(-score)
                weights = weights/weights.sum()
                best = WeightedRandomSampler(
                    weights=weights,
                    num_samples=self.q,
                    replacement=False)
                best = torch.IntTensor(list(best))

        # pop from the back so you don't disrupt the order
        best = best.sort(descending=True).values
        # print(len(self.new), best)
        self.old.extend([self.new.pop(b) for b in best])

    def update_data(self):
        """Update the internal data using old and new."""
        # grab new data
        self.new_data = self.slice_fn(self.data, self.new)

        # grab old data
        self.old_data = self.slice_fn(self.data, self.old)

        # set y_max
        gs, ys = self.old_data
        
        self.y_best = torch.max(ys)

    def run(self, num_rounds=999999, seed=None):
        """Run the model and conduct rounds of acquisition and training.

        Parameters
        ----------
        num_rounds : `int`
             (Default value = 999999)
             Number of rounds.

        seed : `int` or `None`
             (Default value = None)
             Random seed.

        Returns
        -------
        self.old : Resulting indices of acquired candidates.

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
    """Implements active learning experiment with single task target
    with Semi Supervised model."""

    def __init__(self, unlabeled_data=None, *args, **kwargs):
    
        super(SemiSupervisedBayesOptExperiment, self).__init__(*args, **kwargs)
        self.unlabeled_data = unlabeled_data

    def train(self):
        """Train the model with new data."""
        # combine new (unlabeled!) and old (labeled!)
        # Flatten the labeled_data and remove labels to be ready
        semi_supervised_data = pinot.data.utils.prepare_semi_supervised_data(
            self.flatten_data(self.new_data), self.flatten_data(self.old_data)
        )
        
        # Combine this also with the background unlabeled data (if any)
        if self.unlabeled_data:
            semi_supervised_data = pinot.data.utils.prepare_semi_supervised_data(
                self.unlabeled_data, semi_supervised_data
            )

        batched_semi_supervised_data = pinot.data.utils.batch(
            semi_supervised_data, batch_size=len(semi_supervised_data)
        )

        # reset
        self.reset_net()

        # Compute the unsupervised scaling constant and reset it
        # as the number of labeled data points change after every epoch
        if self.unlabeled_data:
            unsup_scale = float(len(self.old_data))/(len(self.new_data) + len(self.unlabeled_data))
        else:
            unsup_scale = float(len(self.old_data))/len(self.new_data)
        
        # Update the unsupervised scale constant of SemiSupervisedNet
        self.net.unsup_scale = unsup_scale

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
        """

        Parameters
        ----------
        data :


        Returns
        -------

        """
        gs, ys = data
        # if gs.batch_size > 1:
        gs = dgl.unbatch(gs)
        ys = list(torch.unbind(ys))

        flattened_data = list(zip(gs, ys))
        return flattened_data
