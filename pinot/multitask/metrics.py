""" Metrics to evaluate and train models.

"""
# =============================================================================
# IMPORTS
# =============================================================================
import dgl
import torch
import pinot
<<<<<<< Updated upstream
=======
from scipy.stats import pearsonr as pr
>>>>>>> Stashed changes

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def _independent(distribution):
    return torch.distributions.normal.Normal(
        loc=distribution.mean.flatten(),
        scale=distribution.variance.pow(0.5).flatten()
    )

# =============================================================================
# MODULE FUNCTIONS
# =============================================================================

def _mse(y, y_hat):
    return torch.nn.functional.mse_loss(y, y_hat)


def mse(net, g, y, l, sampler=None):
	task_metrics = {}
	for task, mask in enumerate(l.T):
	    y_hat_task = net.condition(g, sampler=sampler).mean.cpu()[mask]
	    y_task = y.cpu()[mask]

	    # gp
	    if y_hat_task.dim() == 1:
	        y_hat_task = y_hat_task.unsqueeze(1)
        task_metrics[task] = _mse(y_task, y_hat_task)
    return task_metrics


def _rmse(y, y_hat):
    assert y.numel() == y_hat.numel()
    return torch.sqrt(
        torch.nn.functional.mse_loss(y.flatten(), y_hat.flatten())
    )


def rmse(net, g, y, l, sampler=None):
	task_metrics = {}
	for task, mask in enumerate(l.T):
	    y_hat_task = net.condition(g, sampler=sampler).mean.cpu()[mask]
	    y_task = y.cpu()[mask]

	    # gp
	    if y_hat_task.dim() == 1:
	        y_hat_task = y_hat_task.unsqueeze(1)
        task_metrics[task] = _rmse(y_task, y_hat_task)
    return task_metrics


def _r2(y, y_hat):
    ss_tot = (y - y.mean()).pow(2).sum()
    ss_res = (y_hat - y).pow(2).sum()
    return 1 - torch.div(ss_res, ss_tot)

def r2(net, g, y, l, sampler=None):
    task_metrics = {}
    
    for task, mask in enumerate(l.T):
        y_hat_task = net.condition(g, task, sampler=sampler).mean.cpu()[mask]
        y_task = y.cpu()[mask]

        if y_hat_task.dim() == 1:
            y_hat_task = y_hat_task.unsqueeze(1)

        task_metrics[task] = _r2(y_task, y_hat_task)
    return task_metrics


def log_sigma(net, g, y, sampler=None):
    return net.log_sigma


def pearsonr(net, g, y, l, sampler=None):
    task_metrics = {}

    for task, mask in enumerate(l.T):
	    y_hat_task = net.condition(g, task, sampler=sampler).mean.detach().cpu()[mask]
	    y_task = y.detach().cpu()[mask]
	    
	    result = pr(y_task.flatten().numpy(),
	    			y_hat_task.flatten().numpy())

	    correlation, _ = result
	    task_metrics[task] = torch.Tensor([correlation])[0]
    return task_metrics


def log_sigma(net, g, y, l, sampler=None):
    return net.log_sigma

def avg_nll(net, g, y, l, sampler=None):
    # make the predictive distribution
    distributions = net.condition_train(g, sampler=sampler)

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