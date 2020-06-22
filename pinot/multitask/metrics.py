""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
from scipy.stats import pearsonr as pr
from sklearn.metrics import r2_score, mean_squared_error

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    return torch.distributions.normal.Normal(
        loc=distribution.mean.flatten(),
        scale=distribution.variance.pow(0.5).flatten()
    )

_convert = lambda x: x.detach().cpu().numpy()


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def mse(net, g, y):
    task_metrics = {}

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = mean_squared_error(y_task, y_hat_task)
    return task_metrics

def rmse(net, g, y):
    task_metrics = {}

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = mean_squared_error(y_task, y_hat_task).sqrt()
    return task_metrics


def r2(net, g, y):
    task_metrics = {}

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = r2_score(y_task, y_hat_task)
    return task_metrics

def pearson(net, g, y):
    task_metrics = {}
    
    l = net._generate_mask(y)
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = pr(y_task, y_hat_task)[0]
    return task_metrics

def log_sigma(net, g, y):
    return net.log_sigma


def avg_nll(net, g, y, l, sampler=None):
    # make the predictive distribution
    distributions = net.condition_train(g)

    # make independent
    distributions = [_independent(d) for d in distributions]
    
    task_metrics = {}

    for task, mask in enumerate(l.T):
        # # ensure `y` is one-dimensional
        # assert ((y.dim() == 2 and y.shape[-1] == 1) or 
        #     (y.dim() == 1)
        # )

        # create dummy ys if unlabeled
        y_dummy = torch.zeros(y.shape[0])

        if torch.cuda.is_available():
            y_dummy = y_dummy.to(torch.device('cuda:0'))

        y_dummy[mask] = y[mask, task]

        # calculate the log_prob
        log_prob = distributions[task].log_prob(y_dummy.flatten()).mean()
        task_metrics[task] = -log_prob
    return task_metrics