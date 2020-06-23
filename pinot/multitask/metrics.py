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
def _independent(distribution, mask):
    return torch.distributions.normal.Normal(
        loc=distribution.mean[mask].flatten(),
        scale=distribution.variance[mask].pow(0.5).flatten()
    )

_convert = lambda x: x.detach().cpu().numpy()


# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def mse(net, g, y, **kwargs):
    task_metrics = torch.zeros(y.size(1))

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = mean_squared_error(y_task, y_hat_task)
    return task_metrics

def rmse(net, g, y, **kwargs):
    task_metrics = torch.zeros(y.size(1))

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = torch.sqrt(torch.Tensor([mean_squared_error(y_task, y_hat_task)]))
    return task_metrics


def r2(net, g, y, **kwargs):
    task_metrics = torch.zeros(y.size(1))

    l = net._generate_mask(y)    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = r2_score(y_task, y_hat_task)
    return task_metrics

def pearson(net, g, y, **kwargs):
    task_metrics = torch.zeros(y.size(1))
    
    l = net._generate_mask(y)
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task).mean[mask]
        y_task = y[:, task][mask]
        y_hat_task, y_task = _convert(y_hat_task), _convert(y_task)
        task_metrics[task] = pr(y_task, y_hat_task)[0]
    return task_metrics

def avg_nll(net, g, y, **kwargs):
    task_metrics = torch.zeros(y.size(1))
    
    l = net._generate_mask(y)
    for task, mask in enumerate(l.T):
        distribution = net.condition(g, task)
        distribution = _independent(distribution, mask)
        y_task = y[:, task][mask]
        
        # calculate the log_prob
        log_prob = distribution.log_prob(y_task.flatten()).mean()
        task_metrics[task] = -log_prob
    return task_metrics

def log_sigma(net, g, y, **kwargs):
    return net.log_sigma