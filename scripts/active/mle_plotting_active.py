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
    ds = [tuple([i.to(torch.device('cuda:0')) for i in ds[0]])]

    # get results for each trial
    results = defaultdict(dict)
    final_results = run_trials(results, ds, trial_settings)

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

    best_df = cumulative_regret(ds, best_df)

    return best_df

def run_trials(results, ds, trial_settings):
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
    ys = ds[0][1]
    actual_sol = torch.max(ys).item()
    
    # acquistion functions to be tested
    acq_fns = {'Expected Improvement': pinot.active.acquisition.expected_improvement,
               'Probability of Improvement': pinot.active.acquisition.probability_of_improvement,
               'Upper Confidence Bound': pinot.active.acquisition.upper_confidence_bound,
               # 'Uncertainty': pinot.active.acquisition.uncertainty,
               # 'Random': pinot.active.acquisition.random
               }

    for acq_name, acq_fn in acq_fns.items():
        print(acq_name)
        for i in range(trial_settings['num_trials']):
            print(i)

            # make fresh net and optimizer
            net = get_net(trial_settings).to(torch.device('cuda:0'))

            optimizer = pinot.app.utils.optimizer_translation(
                opt_string=trial_settings['optimizer_name'],
                lr=trial_settings['lr'],
                weight_decay=0.01,
                kl_loss_scaling=1.0/float(len(ds[0][1]))
                )

            # instantiate experiment
            bo = pinot.active.experiment.SingleTaskBayesianOptimizationExperiment(
                        net=net,
                        data=ds[0],
                        optimizer=optimizer(net),
                        acquisition=acq_fn,
                        n_epochs_training=10,
                        k=trial_settings['k'],
                        slice_fn = pinot.active.experiment._slice_fn_tuple,
                        collate_fn = pinot.active.experiment._collate_fn_graph
            )

            # run experiment
            x = bo.run(limit=trial_settings['limit'])

            # record results; pad if experiment stopped early
            # candidates_acquired = limit + 1 because we begin with a blind pick
            results_shape = trial_settings['limit'] * trial_settings['k'] + 1
            results_data = actual_sol*np.ones(results_shape)
            results_data[:len(x)] = np.maximum.accumulate(ys[x].cpu().squeeze())
            print(len(x), results_data[-1])

            results[acq_name][i] = results_data
    
    return results

def cumulative_regret(ds, best_df):
    """
    Compute the cumulative regret across each trial.
    """
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

def get_net(trial_settings):
    """
    Retrive GP using representation provided in args.
    """
    layer = pinot.representation.dgl_legacy.gn(model_name=trial_settings['layer'])

    net_representation = pinot.representation.Sequential(
        layer=layer,
        config=trial_settings['config'])

    net = pinot.Net(net_representation)
    
    return net

if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--representation', type=str)
    parser.add_argument('--num_trials', type=int, default=2)
    parser.add_argument('--limit', type=int, default=2)
    parser.add_argument('--optimizer', type=str, default='Adam')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--k', type=int, default=1)

    args = parser.parse_args()

    trial_settings = {'layer': args.representation,
                      'config': [32, 'tanh', 32, 'tanh', 32, 'tanh'],
                      'data': 'moonshot',
                      'num_trials': args.num_trials,
                      'limit': args.limit,
                      'optimizer_name': args.optimizer,
                      'lr': args.lr,
                      'k': args.k}

    best_df = generate_data(trial_settings)

    # save to disk
    filename = f'best_{args.representation}_{args.optimizer}_greedy_k{args.k}.csv'
    best_df.to_csv(filename)
