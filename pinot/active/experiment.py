# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import dgl
import pinot

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    return torch.distributions.normal.Normal(
        distribution.mean,
        distribution.variance.pow(0.5))

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

class SingleTaskBayesianOptimizationExperiment(ActiveLearningExperiment):
    """ Implements active learning experiment with single task target.

    """
    def __init__(
            self,
            net,
            data,
            acquisition,
            optimizer,
            n_epochs_training=100,
            workup=_independent,
            slice_fn=_slice_fn_tensor,
            collate_fn=_collate_fn_tensor,
            net_state_dict=None
        ):

        super(SingleTaskBayesianOptimizationExperiment, self).__init__()

        # model
        self.net = net
        self.optimizer = optimizer
        self.n_epochs_training = n_epochs_training

        # data            
        self.data = data
        self.old = []
        if isinstance(data, tuple):
            self.new = list(range(len(data[1])))
        else:
            self.new = list(range(len(data)))

        # acquisition
        self.acquisition = acquisition

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

    def acquire(self):
        """ Acquire new training data.

        """
        # set net to eval
        self.net.eval()

        # grab new data
        new_data = self.slice_fn(self.data, self.new)

        # split input target
        gs, ys = new_data

        # get the predictive distribution
        # TODO:
        # write API for sampler
        distribution = self.net.condition(gs)

        # workup
        distribution = self.workup(distribution)

        # get score
        # NOTE: y best here doesn't change argmax
        score = self.acquisition(distribution)

        # argmax
        best = torch.argmax(score)

        # grab
        self.old.append(self.new.pop(best))

    def train(self):
        """ Train the model with new data.

        """
        # reset
        self.reset_net()

        # set to train status
        self.net.train()

        # grab old data
        old_data = self.slice_fn(self.data, self.old)

        # train the model
        self.net = pinot.app.experiment.Train(
            data=[old_data],
            optimizer=self.optimizer,
            n_epochs=self.n_epochs_training,
            net=self.net,
            record_interval=999999).train()

    def run(self, limit=999999):
        """ Run the model.

        Parameters
        ----------
        limit : int, default=99999

        """
        idx = 0
        self.blind_pick()

        while idx < limit and len(self.new) > 0:
            self.train()
            self.acquire()
            idx += 1

        return self.old
