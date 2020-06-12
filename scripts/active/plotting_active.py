import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import pinot
import pinot.active


######################
# Function definitions

def generate_data(trial_settings):
    """
    Performs experiment loops.
    """

    # Load and batch data
    ds = getattr(pinot.data, trial_settings['data'])()
    ds = pinot.data.utils.batch(ds, len(ds), seed=None)

    # get results for each trial
    results = defaultdict(dict)
    final_results = run_trials(results, ds,
        num_trials=trial_settings['num_trials'],
        limit=trial_settings['limit'])

    # create pandas dataframe to play nice with seaborn
    best_df = pd.DataFrame.from_records(
        [
            (acq_fn, trial, step, best)
            for acq_fn, trial_dict in dict(final_results).items()
            for trial, best_history in trial_dict.items()
            for step, best in enumerate(best_history)
        ],
        columns=['Acquisition Function', 'Trial', 'Step', 'Best Solubility']
    )

    # compute cumulative regret
    actual_best = max(ds[0][1].squeeze())
    best_df['Regret'] = actual_best.item() - best_df['Best Solubility']

    cum_regret_all = []
    for acq_fn in best_df['Acquisition Function'].unique():
        sums = best_df[best_df['Acquisition Function'] == acq_fn].groupby(['Step', 'Trial']).sum()
        cum_regret = sums['Regret'].unstack('Trial').cumsum()
        cum_regret_all.append(cum_regret.melt().values[:,1])
    cum_regret_all = np.concatenate(cum_regret_all)
    best_df['Cumulative Regret'] = cum_regret_all

    return best_df

def run_trials(results, ds, num_trials=1, limit=5):
    """
    Plot the results of an active training loop
    
    Parameters
    ----------
    results : defaultdict of dict
        Empty dict in which to store results.
    ds : list of tuple
        Output of `pinot.data` and batched. Contains DGLGraph gs and Tensor ys.
    num_trials : int
        number of times to run each acquisition function
    limit : int
        Number of runs of bayesian optimization.
    Returns
    -------
    results
    """
    
    # get the real result
    actual_sol = ds[0][1].squeeze().numpy()
    
    # acquistion functions to be tested
    acq_fns = {'random': pinot.active.acquisition.random,
               'expected improvement': pinot.active.acquisition.expected_improvement,
               'probability of improvement': pinot.active.acquisition.probability_of_improvement,
               'upper confidence bound': pinot.active.acquisition.upper_confidence_bound}

    for acq_fn in acq_fns:

        for i in range(num_trials):

            # make fresh net
            net = get_gpr(trial_settings)
            bo = pinot.active.experiment.SingleTaskBayesianOptimizationExperiment(
                        net=net,
                        data=ds[0],
                        optimizer=torch.optim.Adam(net.parameters(), 1e-3),
                        acquisition=acq_fns[acq_fn],
                        n_epochs_training=10,
                        slice_fn = pinot.active.experiment._slice_fn_tuple,
                        collate_fn = pinot.active.experiment._collate_fn_graph
            )

            # run experiment
            x = bo.run(limit=limit)
            results[acq_fn][i] = get_best_history(bo, actual_sol)
    
    return results

def get_best_history(bo, actual_sol):
    """
    Finds the best solubility of candidates chosen at time step.
    
    Returns
    -------
    best_history : list of float
        The best solubility seen so far.
    """
    best_history = []
    best_so_far = -np.inf
    for candidate in bo.old:
        if actual_sol[candidate] > best_so_far:
            best_so_far = actual_sol[candidate]
        best_history.append(best_so_far)
    return best_history

def get_gpr(trial_settings):
    """
    Retrive GP using representation provided in args.
    """
    layer = pinot.representation.dgl_legacy.gn(model_name=trial_settings['layer'])

    net_representation = pinot.representation.Sequential(
        layer=layer,
        config=trial_settings['config'])

    kernel = pinot.inference.gp.kernels.deep_kernel.DeepKernel(
            representation=net_representation,
            base_kernel=pinot.inference.gp.kernels.rbf.RBF())

    gpr = pinot.inference.gp.gpr.exact_gpr.ExactGPR(
            kernel)
    return gpr


# Running functions
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--representation', type=str)
parser.add_argument('--num_trials', type=int, default=10)
parser.add_argument('--limit', type=int, default=100)

args = parser.parse_args()

trial_settings = {'layer': args.representation,
                  'config': [32, 'tanh', 32, 'tanh', 32, 'tanh'],
                  'data': 'esol',
                   'num_trials': args.num_trials,
                   'limit': args.limit}

best_df = generate_data(trial_settings)

# save to disk
best_df.to_csv(f'best_{args.representation}.csv')