# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import dgl
import copy
import pinot
from collections import OrderedDict
from pinot.metrics import _independent

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

    `tensor_slices` : `torch.Tensor`
        Sliced and stacked tensor.
    """
    inputs, outputs = x

    if isinstance(inputs, dgl.DGLHeteroGraph):
        _slice_fn_input = _slice_fn_graph
    else:
        _slice_fn_input = _slice_fn_tensor

    input_slices = _slice_fn_input(inputs, idxs)
    output_slices = _slice_fn_tensor(outputs, idxs)
    return input_slices, output_slices


# =============================================================================
# MODULE CLASSES
# =============================================================================
class ActiveLearningExperiment(torch.nn.Module, abc.ABC):
    """Implements active learning experiment base class."""

    def __init__(self):
        super(ActiveLearningExperiment, self).__init__()

    @abc.abstractmethod
    def train_round(self, *args, **kwargs):
        """ Train the model. """
        raise NotImplementedError

    @abc.abstractmethod
    def acquire_round(self, *args, **kwargs):
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

    q : `int`
        Number of acquired candidates in each round.

    acquisitions : `list`
        Provided list of acquisitions to hard-code at each round.

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
        utility_func,
        optimizer,
        num_epochs=100,
        q=1,
        num_samples=1000,
        early_stopping=True,
        workup=_independent,
        slice_fn=_slice_fn_tensor,
        collate_fn=_collate_fn_tensor,
        net_state_dict=None,
        acquisitions_preset=None,
        annealing=1.0,
        train_class=pinot.app.experiment.Train,
        update_representation_interval=20
    ):

        super(BayesOptExperiment, self).__init__()

        # model
        self.net = net
        self.num_epochs = num_epochs
        
        # make optimizers
        self.optimizer = optimizer(net)
        # TODO: check that it is actually only picking the relevant parameters
        self.optimizer_output_regressor = optimizer(net.output_regressor)
        self.update_representation_interval = update_representation_interval

        # data
        self.data = data
        self.seen_index = []
        self.unseen_index = list(range(len(data)))
        
        # acquisition
        self.utility_func = utility_func
        self.q = q

        # early stopping
        self.early_stopping = early_stopping
        self.best_possible = self.get_y_best(data, self.unseen_index)

        # bookkeeping
        self.workup = workup
        self.slice_fn = slice_fn
        self.collate_fn = collate_fn
        self.net_state_dict = net_state_dict
        self.annealing = annealing
        self.train_class = train_class
        self.net_params = {}

        if acquisitions_preset:
            self.acquisitions = acquisitions_preset
        else:
            self.acquisitions = OrderedDict()


    def reset_net(self, net):
        """Reset everything."""
        # TODO:
        # reset optimizer too
        (p.reset_parameters() for _, p in net.named_children())

        if self.net_state_dict is not None:
            net.load_state_dict(self.net_state_dict)

        return net


    def train_round(
        self,
        net,
        data,
        seen_index,
        update_representation=False,
        batch_size=256
    ):
        """Train the model with new data."""

        # obtain seen data for training
        train_data = data[seen_index].batch(
            batch_size, partial_batch=True
        )

        # optimize full net w/ graphs
        if update_representation:
            
            # reset net - TEST EMPIRICALLY
            net = self.reset_net(net)

            # TODO: ***reset optimizer***
            num_epochs = self.num_epochs
            optimizer = self.optimizer
            qsar_model = net

        # optimize regressor-only w/ hidden rep
        else:

            # no reset, shallow training
            num_epochs = 40 # 40
            optimizer = self.optimizer_output_regressor
            qsar_model = net.output_regressor

        # train the model
        qsar_model.train()

        tr = self.train_class(
            net=qsar_model,
            data=train_data,
            optimizer=optimizer,
            n_epochs=num_epochs,
            annealing=self.annealing,
            record_interval=999999,
        ).train()

        # update net
        if update_representation:
            net = tr.net
        else:
            net.output_regressor = tr.net

        return net


    def get_y_best(self, data, index):
        """Get maximum value among an index over data."""
        
        # grab old data
        data_subset = data[index]

        # gather outputs
        ys = [d[1] for d in data_subset]

        # find max output
        y_best = max(ys)
        
        return y_best


    def get_hidden_inputs(self, net, data):
        """ Helper function for getting hidden inputs if updating rep
        """
        # modify net or data if not updating representation
        get_h = lambda x: (net.representation(x[0]).detach(), x[1])
        qsar_model = net.output_regressor
        data_idx = data.apply(get_h)
        return qsar_model, data_idx


    def _acquire(
        self,
        net,
        candidate_data,
        utility_func,
        q=1,
        y_best=float("-Inf")
    ):
        """Acquire new training data."""
        
        # set net to eval
        net.eval()

        # acquire no more points than are remaining
        if self.q <= len(candidate_data):

            # acquire pending points
            pending_pts = utility_func(net, candidate_data, q=q, y_best=y_best)
    
            # pop from the back so you don't disrupt the order
            pending_pts = pending_pts.sort(descending=True).values

        else:

            # set pending points to all remaining data
            n_obs = len(candidate_data)
            pending_pts = torch.range(n_obs-1, 0, step=-1).long()

        return pending_pts


    def acquire_round(
        self,
        qsar_model,
        data,
        utility_func,
        seen_index,
        unseen_index,
        q=1,
        y_best=float("-Inf"),
        acquisitions_preset=None,
    ):
        """ Acquires pending points given what has been seen
        """
        # preset acquisition policy
        if acquisitions_preset is not None:
            seen_index, unseen_index = acquisitions_preset

        # autonomous acquisition policy
        else:

            # obtain unseen data for acquisitions
            unseen_data = data[unseen_index]
            
            # acquire pending points
            pending_indices = self._acquire(
                qsar_model,
                unseen_data,
                utility_func,
                q=q,
                y_best=y_best
            )

            # update seen and unseen
            seen_index.extend([
                unseen_index.pop(p)
                for p in pending_indices
            ])

        return seen_index, unseen_index


    def run(
        self,
        net=None,
        data=None,
        utility_func=None,
        q=0,
        num_rounds=999999,
        acquisitions=OrderedDict(),
        acquisitions_preset=None
    ):
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
        if not net:
            net = self.net
        if not data:
            data = self.data
        if not utility_func:
            utility_func = self.utility_func
        if not q:
            q = self.q
        seen_index, unseen_index = self.seen_index, self.unseen_index

        # initialize variables
        y_best = -float("Inf")
        qsar_model_idx = net
        data_idx = data
        
        idx = 0
        while idx < num_rounds and unseen_index:

            print(idx)
            
            # early stopping logic
            if self.early_stopping and y_best == self.best_possible:
                break

            # train full net / representation?
            interval_count = idx % self.update_representation_interval
            update_representation = interval_count == 0
            update_next_round = interval_count == (self.update_representation_interval - 1)

            # use random policy in the first round
            if idx == 0:
                utility_func_idx = pinot.active.acquisition.random
            else:
                utility_func_idx = self.utility_func

            # acquire points
            seen_index, unseen_index = self.acquire_round(
                qsar_model_idx,
                data_idx,
                utility_func_idx,
                seen_index,
                unseen_index,
                q=q,
                y_best=y_best,
                acquisitions_preset=acquisitions_preset
            )

            # record net states before training
            self.net_params[idx] = copy.deepcopy(net.state_dict())

            # train
            net = self.train_round(
                net,
                data_idx,
                seen_index,
                update_representation=update_representation
            )

            # get hidden depending on round
            if update_representation and not update_next_round:
                qsar_model_idx, data_idx = self.get_hidden_inputs(net, data)
            elif update_next_round:
                qsar_model_idx, data_idx = net, data

            # record acquisitions + bookkeeping
            y_best = self.get_y_best(data_idx, seen_index)
            acquisitions[idx] = tuple([seen_index.copy(), unseen_index.copy()])
            self.seen_index, self.unseen_index = seen_index, unseen_index

            idx += 1

        return acquisitions










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
            self.flatten_data(self.unseen_data),
            self.flatten_data(self.seen_data),
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
            unsup_scale = float(len(self.seen_data))/(len(self.unseen_data) + len(self.unlabeled_data))
        else:
            unsup_scale = float(len(self.seen_data))/len(self.unseen_data)
        
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