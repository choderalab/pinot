# =============================================================================
# IMPORTS
# =============================================================================
import torch
import abc
import pinot

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    return torch.distributions.normal.Normal(
        distribution.mean,
        distribution.variance.pow(0.5))

def _slice_fn_tensor(data, idxs):
    # data.shape = (N, 2)
    # idx is a list
    data = data[idxs]
    assert data.dim() == 2
    assert data.shape[-1] == 2
    return data[:, 0][:, None], data[:, 1][:, None]

def _slice_fn_graph(data, idx):
    # data is a list
    # idx is a list
    data = [data[idx] for idx in idxs]
    gs, ys = list(zip(*data))
    import dgl
    gs = dgl.batch(gs)
    ys = torch.stack(ys, dim=0)
    return gs, ys

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
        self.new = list(range(len(data)))

        # acquisition
        self.acquisition = acquisition

        # bookkeeping
        self.workup = workup
        self.slice_fn = slice_fn
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

        # slice and collate
        gs, _ = self.slice_fn(self.data, self.new)

        # get the predictive distribution
        # TODO:
        # write API for sampler
        distribution = self.net.condition(gs)

        # workup
        distribution = self.workup(distribution)

        # get score
        # NOTE: y best here doesn't change argmax
        score = self.acquisition(distribution, y_best=self.y_best)

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
        # (N, 2) for tensor
        # N list of 2-tuple for lists
        old_data = self.slice_fn(self.data, self.old)

        # train the model
        self.net = pinot.app.experiment.Train(
            data=[old_data],
            optimizer=self.optimizer,
            n_epochs=self.n_epochs_training,
            net=self.net,
            record_interval=999999).train()

        # grab y_max
        gs, ys = old_data
        self.y_best = torch.max(ys)

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
