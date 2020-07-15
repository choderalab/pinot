import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import pinot
from pinot.generative import SemiSupervisedNet
from pinot.metrics import _independent
from pinot.active.acquisition import thompson_sampling
from pinot.active.biophysical_acquisition import (biophysical_thompson_sampling,
                                                  _sample_and_marginalize_delta_G)


######################
# Utilities

def _get_thompson_values(net, data, q=1):
    """ Gets the value associated with the Thompson Samples.
    """
    # obtain predictive posterior
    gs, ys = data
    distribution = _independent(net.condition(gs))
    
    # obtain samples from posterior
    thetas = distribution.sample((q,)).detach()
    thompson_values = torch.max(thetas, axis=1).values
    return thompson_values


def _get_biophysical_thompson_values(
    net,
    data,
    acq_func,
    q=1,
    concentration=20,
    dG_samples=10):
    """ Generates m Thompson samples and maximizes them.
    """        
    # fill batch
    thompson_values = []
    for _ in range(q):

        # for each thompson sample,
        # get max, marginalizing across all distributions
        thompson_value = _sample_and_marginalize_delta_G(
            net,
            data,
            concentration=concentration,
            dG_samples=dG_samples,
            n_samples=1
        ).max().item()
        thompson_values.append(thompson_value)

    # convert to tensor
    thompson_values = torch.Tensor(thompson_values)
    return thompson_values


######################
# Function definitions

class TSBayesOptExperiment(pinot.active.experiment.BayesOptExperiment):
    """ Performs Thompson Sampling each loop.
    """
    def __init__(
        self,
        num_thompson_samples=1000,
        *args,
        **kwargs
        ):

        super(TSBayesOptExperiment, self).__init__(*args, **kwargs)

        self.early_stopping=False
        self.num_thompson_samples = num_thompson_samples


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

        while idx < num_rounds and len(self.unseen) > 0:
            
            self.train()
            self.thompson_sample(idx, num_rounds, num_samples=self.num_thompson_samples)
            self.acquire()
            self.update_data()

            if self.early_stopping and self.y_best == self.best_possible:
                break

            idx += 1

        return self.seen, self.thompson_samples


    def thompson_sample(self, idx, num_rounds, num_samples=1):
        """ Perform retrospective and prospective Thompson Sampling
            to check model beliefs about y_max.
        """
        def _ts(key, idx, data, num_samples=1):
            """Get Thompson samples.
            """
            if isinstance(self.net.output_regressor, pinot.regressors.BiophysicalRegressor):
                self.thompson_samples[key][idx] = _get_biophysical_thompson_values(self.net, data, q=num_samples)
            else:
                self.thompson_samples[key][idx] = _get_thompson_values(self.net, data, q=num_samples)

        # set net to eval
        self.net.eval()

        if not hasattr(self, 'thompson_samples'):
            self.thompson_samples = {'prospective': torch.Tensor(num_rounds, num_samples),
                                     'retrospective': torch.Tensor(num_rounds, num_samples)}
        
        for key in self.thompson_samples:
            # thompson sampling on UNSEEN data if prospective
            data = self.unseen_data if key == 'prospective' else self.data
            _ts(key, idx, data, num_samples=num_samples)


class TSActivePlot():

    def __init__(self, net, config,
                 lr, optimizer_type,
                 data, acquisition, num_samples, num_thompson_samples, q,
                 device, num_trials, num_rounds, num_epochs):

        # net config
        self.net = net
        self.config = config

        # optimizer config
        self.lr = lr
        self.optimizer_type = optimizer_type

        # experiment config
        self.data = data
        self.acquisition = acquisition
        self.num_samples = num_samples
        self.num_thompson_samples = num_thompson_samples
        self.q = q
        self.train = pinot.app.experiment.Train

        # housekeeping
        self.device = torch.device(device)
        self.num_trials = num_trials
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs

        # recording data
        self.results = []
        self.prospective_ts = []
        self.retrospective_ts = []


    def generate(self):
        """
        Performs experiment loops.
        """
        ds = self.generate_data()

        # get results for each trial
        final_results, prospective_ts, retrospective_ts = self.run_trials(ds)

        # create pandas dataframe to play nice with seaborn
        best_df = pd.DataFrame(final_results)
        pro_ts_df = pd.DataFrame(prospective_ts)
        retro_ts_df = pd.DataFrame(retrospective_ts)

        return best_df, pro_ts_df, retro_ts_df


    def run_trials(self, ds):
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
        # get the real solution
        ds = self.generate_data()
        acq_fn = self.get_acquisition()

        # acquistion functions to be tested
        for i in range(self.num_trials):

            print(i)

            # make fresh net and optimizer
            net = self.get_net().to(self.device)

            optimizer = pinot.app.utils.optimizer_translation(
                opt_string=self.optimizer_type,
                lr=self.lr,
                weight_decay=0.01,
                kl_loss_scaling=1.0/float(len(ds[0][1]))
                )

            # instantiate experiment
            self.bo = TSBayesOptExperiment(
                net=net,
                data=ds[0],
                optimizer=optimizer,
                acquisition=acq_fn,
                num_epochs=self.num_epochs,
                num_thompson_samples=self.num_thompson_samples,
                q=self.q,
                slice_fn=pinot.active.experiment._slice_fn_tuple, # pinot.active.
                collate_fn=pinot.active.experiment._collate_fn_graph, # pinot.active.
                train_class=self.train
            )

            # run experiment
            x, self.ts = self.bo.run(num_rounds=self.num_rounds)
            
            # record results
            self.results = self.process_results(
                x[1:], # ignore first random pick
                ds,
                i
            )

            self.prospective_ts = self.process_thompson_samples(
                self.prospective_ts,
                self.ts['prospective'],
                i
            )

            self.retrospective_ts = self.process_thompson_samples(
                self.retrospective_ts,
                self.ts['retrospective'],
                i
            )

        return self.results, self.prospective_ts, self.retrospective_ts


    def process_thompson_samples(self, ts_list, ts, i):
        """ Processes the output of BayesOptExperiment.

        Parameters
        ----------
        thompson_samples : dict
            Output of `TSBayesOptExperiment` object.
            Keys are 'prospective' and 'retrospective'.
            Values are torch.Tensors of indices, corresponding to Thompson samples.
        ts : tuple
            Thompson sampling values
        i : int
            Index of loop

        Returns
        -------
        self.thompson_samples : dict
            Processed data
        """
        # record results
        for step, step_ts in enumerate(ts):
            
            for ts_index, t in enumerate(step_ts):
                
                ts_list.append({'Acquisition Function': self.acquisition,
                                'Trial': i,
                                'Round': step,
                                'TS Index': ts_index,
                                'Value': t.item()})
        return ts_list


    def process_results(self, x, ds, i):
        """ Processes the output of BayesOptExperiment.

        Parameters
        ----------
        x : list of int
            Items chosen by BayesOptExperiment object
        ds : tuple
            Dataset object
        i : int
            Index of loop

        Returns
        -------
        self.results : defaultdict
            Processed data
        """
        gs, ys = ds[0]
        actual_sol = torch.max(ys).item()

        # pad if experiment stopped early
        # candidates_acquired = limit + 1 because we begin with a blind pick
        results_size = self.num_rounds * self.q # (actually no +1)

        if self.net == 'multitask':
            results_data = actual_sol*np.ones((results_size, ys.size(1)))
            output = ys[x]
            output[torch.isnan(output)] = -np.inf
        else:
            results_data = actual_sol*np.ones(results_size)
            output = ys[x]

        results_data[:len(x)] = np.maximum.accumulate(output.cpu().squeeze())

        # record results
        for step, result in enumerate(results_data):
            self.results.append({'Acquisition Function': self.acquisition,
                                 'Trial': i, 'Datapoint Acquired': step,
                                 'Round': step // self.q, 'Value': result})
        return self.results


    def generate_data(self):
        """
        Generate data, put on GPU if possible.
        """
        # Load and batch data
        ds = getattr(pinot.data, self.data)()
        ds = pinot.data.utils.batch(ds, len(ds), seed=None)
        ds = [tuple([i.to(self.device) for i in ds[0]])]
        return ds

    def get_acquisition(self):
        """ Retrieve acquisition function and prepare for BO Experiment
        """
        acquisitions = {
            'ExpectedImprovement': pinot.active.acquisition.expected_improvement_analytical,
            'ProbabilityOfImprovement': pinot.active.acquisition.probability_of_improvement,
            'UpperConfidenceBound': pinot.active.acquisition.upper_confidence_bound,
            'Uncertainty': pinot.active.acquisition.uncertainty,
            'Human': pinot.active.acquisition.temporal,
            'Random': pinot.active.acquisition.random,
            'ThompsonSampling': pinot.active.acquisition.thompson_sampling,
        }

        return acquisitions[self.acquisition]


    def get_net(self):
        """
        Retrive GP using representation provided in args.
        """
        representation = pinot.representation.SequentialMix(config=self.config)

        if hasattr(pinot.regressors, self.net):
            output_regressor = getattr(pinot.regressors, self.net)
            net = pinot.Net(
                representation=representation,
                output_regressor_class=output_regressor,
            )

        if self.net == 'semi':
            output_regressor = pinot.regressors.NeuralNetworkRegressor
            net = SemiSupervisedNet(
                representation=representation,
                output_regressor=output_regressor
            )

        elif self.net == 'semi_gp':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor

            net = SemiSupervisedNet(
                representation=representation,
                output_regressor=output_regressor,
            )

        elif self.net == 'multitask':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
            net = pinot.multitask.MultitaskNet(
                representation=representation,
                output_regressor=output_regressor,
            )

        return net


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='ExactGaussianProcessRegressor')
    parser.add_argument('--config', nargs="+", type=str,
        default=["GraphConv", "32", "activation", "tanh",
        "GraphConv", "32", "activation", "tanh",
        "GraphConv", "32", "activation", "tanh"])

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--data', type=str, default='esol')
    parser.add_argument('--acquisition', type=str, default='ExpectedImprovement')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--num_thompson_samples', type=int, default=1000)
    parser.add_argument('--q', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--index_provided', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    plot = TSActivePlot(
        # net config
        net=args.net,
        config=args.config,

        # optimizer config
        optimizer_type=args.optimizer,
        lr=args.lr,

        # experiment config
        data=args.data,
        acquisition=args.acquisition,
        num_samples=args.num_samples,
        num_thompson_samples=args.num_thompson_samples,        
        q=args.q,

        # housekeeping
        device=args.device,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
    )

    best_df = plot.generate()
    representation = args.config[0]

    # save to disk
    if args.index_provided:
        filename = f'{args.net}_{representation}_{args.optimizer}_{args.data}_{args.acquisition}_q{args.q}_{args.index}.csv'
    best_df.to_csv(filename)