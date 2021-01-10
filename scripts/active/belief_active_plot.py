import warnings
warnings.filterwarnings("ignore")

import copy
import math
from contextlib import suppress
from collections import defaultdict, OrderedDict

import toolz
import torch
import numpy as np
import pandas as pd

import pinot
from pinot.generative import SemiSupervisedNet
from pinot.metrics import _independent
from pinot.active.acquisition import thompson_sampling, _pi, _ei_analytical, _ucb
from pinot.active.biophysical_acquisition import (biophysical_thompson_sampling,
                                                  _sample_and_marginalize_delta_G)

######################
# Utilities

def _get_thompson_values(net, data, q=1):
    """ Gets the value associated with the Thompson Samples.
    """
    # unpack data
    gs, _ = data

    # get predictive posterior
    distribution = _independent(net.condition(gs))
    
    # obtain samples from posterior
    thetas = distribution.sample((q,)).detach()

    # find the max of each sample
    thompson_values, thompson_indices = torch.max(thetas, axis=1)

    return thompson_values, thompson_indices


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


def thompson_sample(net, data, unseen, num_samples=1, **kwargs):
    """ Perform retrospective and prospective Thompson Sampling
        to check model beliefs about y_max.

    Returns
    -------
    thompson_samples : dict
        Dictionary with keys 'prospective' and 'retrospective'
        Values are tensors.
    """
    def _ts(net, data, num_samples=1):
        """ Get Thompson samples.
        """
        if isinstance(net.output_regressor, pinot.regressors.BiophysicalRegressor):
            ts_values = _get_biophysical_thompson_values(net, data, q=num_samples)
        else:
            ts_values, ts_indices = _get_thompson_values(net, data, q=num_samples)
        return ts_values, ts_indices

    # save log sigma
    sigma = torch.exp(net.output_regressor.log_sigma)

    # construct measurement error distribution
    epsilon = torch.distributions.Normal(0., sigma)

    # make predictive posterior noiseless by setting sigma to 0
    zero = torch.tensor(-10.)
    if torch.cuda.is_available():
        zero = zero.cuda()
    net.output_regressor.log_sigma = torch.nn.Parameter(zero)

    # set net to eval
    net.eval()

    # initialize thompson sampling dict
    thompson_samples_values = {}
    thompson_samples_indices = {}
    
    # get prospective thompson samples
    if unseen:

        # prospective evaluates only unseen data
        unseen_data = pinot.active.experiment._slice_fn_tuple(data, unseen)
        ts_values, ts_indices = _ts(
            net,
            unseen_data,
            num_samples=num_samples
        )
        thompson_samples_indices['prospective'] = ts_indices
        thompson_samples_values['prospective'] = ts_values + epsilon.sample((num_samples,))

    # get retrospective thompson samples on all data
    ts_values, ts_indices = _ts(
        net,
        data,
        num_samples=num_samples
    )
    thompson_samples_indices['retrospective'] = ts_indices
    thompson_samples_values['retrospective'] = ts_values + epsilon.sample((num_samples,))
    
    return thompson_samples_indices, thompson_samples_values


def _max_utility(net, data, unseen, utility_func):
    """ Finds max of the beliefs of a network according to some utility function
    """
    # unpack data
    gs, ys = data

    # get y_best
    seen = [s for s in range(len(ys)) if s not in unseen]
    y_best_round = ys[seen].max().item()
    y_best_global = ys.max().item()

    # set net to eval
    net.eval()

    # get predictive posterior
    distribution = _independent(net.condition(gs))

    # initialize beliefs dict
    beliefs = {}

    # get prospective beliefs
    if unseen:

        # prospective evaluates only unseen data
        unseen_data = pinot.active.experiment._slice_fn_tuple(data, unseen)
        beliefs['prospective'] = torch.max(
            utility_func(
                distribution, y_best=y_best_round,
            )
        ).detach().unsqueeze(0)

    # get retrospective thompson samples on all data
    beliefs['retrospective'] = torch.max(
        utility_func(
            distribution, y_best=y_best_global,
        )
    ).detach().unsqueeze(0)

    return beliefs


def max_probability_of_improvement(net, data, unseen, **kwargs):
    """ Computes the belief about the max using PI utility function.
    """
    beliefs = _max_utility(net, data, unseen, _pi)
    return beliefs


def max_upper_confidence_bound(net, data, unseen, **kwargs):
    """ Computes the belief about the max using UCB utility function.
    """
    beliefs = _max_utility(net, data, unseen, _ucb)
    return beliefs


def max_expected_improvement(net, data, unseen, **kwargs):
    """ Computes the belief about the max using EI utility function.
    """
    beliefs = _max_utility(net, data, unseen, _ei_analytical)
    return beliefs


######################
# Function definitions

class BeliefActivePlot():

    def __init__(
        self, net, architecture,
        n_inducing_points, annealing,
        lr, optimizer_type,
        data, sample_frac,
        acquisition, num_samples, num_thompson_samples, q,
        beliefs, early_stopping,
        device, num_trials, num_rounds, num_epochs):

        # net config
        self.net = net
        self.architecture = architecture
        self.n_inducing_points = n_inducing_points
        self.annealing = annealing

        # optimizer config
        self.lr = lr
        self.optimizer_type = optimizer_type

        # data config
        self.data = data
        self.sample_frac = sample_frac
        
        # experiment config
        self.acquisition = acquisition
        self.num_samples = num_samples
        self.num_thompson_samples = num_thompson_samples
        self.q = q
        self.train = pinot.app.experiment.Train
        self.early_stopping = early_stopping

        # beliefs
        self.belief_functions = beliefs
        self.belief_functions_dict = {
            'ThompsonSampling': thompson_sample,
            'MaxProbabilityOfImprovement': max_probability_of_improvement,
            'MaxUCB': max_upper_confidence_bound,
            'MaxExpectedImprovement': max_expected_improvement,
        }

        # housekeeping
        self.device = torch.device(device)
        self.num_trials = num_trials
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs

        # recording data
        self.results = []
        self.prospective_beliefs = []
        self.retrospective_beliefs = []


    def generate(self):
        """
        Performs experiment loops.
        """
        ds = self.generate_data()

        # get results for each trial
        final_results, prospective_beliefs, retrospective_beliefs = self.run_trials(ds)

        # create pandas dataframe to play nice with seaborn
        best_df = pd.DataFrame(final_results)
        pro_df = pd.DataFrame(prospective_beliefs)
        retro_df = pd.DataFrame(retrospective_beliefs)

        return best_df, pro_df, retro_df


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
        # get the data
        ds, ds_dates = self.generate_data()

        # prepare autonomous acquisition functions
        acq_fn = self.get_acquisition_fn()
        
        # get acquisitions if using human function
        if self.acquisition == 'Human' and ds_dates:
            acquisitions_preset = self.gather_acquisitions(ds_dates)
        else:
            acquisitions_preset = None

        for i in range(self.num_trials):

            print(i)

            # make fresh net and optimizer
            net = self.get_net().to(self.device)

            optimizer = pinot.app.utils.optimizer_translation(
                opt_string=self.optimizer_type,
                lr=self.lr,
                weight_decay=0.01,
                kl_loss_scaling=1.0/float(len(ds[1]))
            )

            # instantiate experiment
            self.bo = pinot.active.experiment.BayesOptExperiment(
                net=net,
                data=ds,
                optimizer=optimizer,
                acquisition=acq_fn,
                num_epochs=self.num_epochs,
                early_stopping=self.early_stopping,
                q=self.q,
                acquisitions_preset=acquisitions_preset,
                slice_fn=pinot.active.experiment._slice_fn_tuple,
                collate_fn=pinot.active.experiment._collate_fn_graph,
                annealing=self.annealing,
                train_class=self.train
            )

            # run experiment
            acquisitions = self.bo.run(num_rounds=self.num_rounds)

            # process acquisitions into pandas-friendly format
            self.results = self.process_results(acquisitions, ds, i)
            
            # get beliefs
            if self.belief_functions:

                self.beliefs = self.get_beliefs(net, ds, acquisitions, methods=self.belief_functions)

                # generate long-form records for pandas
                self.prospective_beliefs.extend(
                    self.process_beliefs(self.beliefs['prospective'], i)
                )

                # generate long-form records for pandas
                self.retrospective_beliefs.extend(
                    self.process_beliefs(self.beliefs['retrospective'], i)
                )

        return self.results, self.prospective_beliefs, self.retrospective_beliefs


    def get_beliefs(self, net, ds, acquisitions, methods=['ThompsonSampling']):
        """ Gets Thompson samples at each round

        Parameters
        ----------
        net : pinot.Net
            The network used for Bayesian optimization
        
        ds : pinot.data.Dataset
            The dataset used
        
        acquisitions : list of dict
            Choice of acquired and unacquired points at each round

        methods : list of str
            The method used to estimate model beliefs 

        Returns
        -------
        beliefs : dict
            Keys are 'prospective' and 'retrospective'
            Values are dictionaries of beliefs
        
        """        
        # loop through rounds
        round_beliefs = []
        for idx, state in self.bo.states.items():

            # unpack acquisitions
            seen, unseen = acquisitions[idx]

            # load network states
            net.load_state_dict(state)

            # set _x_tr and _y_tr if exact GP
            if isinstance(
                net.output_regressor,
                pinot.regressors.ExactGaussianProcessRegressor
            ):
                seen_gs, seen_ys = pinot.active.experiment._slice_fn_tuple(ds, seen)
                net.g_last, net.y_last = seen_gs, seen_ys

            # gather pro and retro beliefs for each belief function
            for belief_name, belief_function in self.belief_functions_dict.items():
                if belief_name in methods:
                    round_belief_index, round_belief = belief_function(
                        net, ds, unseen,
                        num_samples=self.num_thompson_samples
                    )
                    round_beliefs.append(
                        {belief_name: {'index': round_belief_index, 'belief': round_belief}}
                    )

        # convert from list of dicts to dict of lists
        beliefs = {'prospective': defaultdict(list), 'retrospective': defaultdict(list)}
        print(round_beliefs[-1])
        for round_record in round_beliefs:
            for m in methods:
                for direction in beliefs.keys():
                    with suppress(KeyError):
                        index, belief = (
                            round_record[m]['index'][direction],
                            round_record[m]['belief'][direction]
                        )
                        beliefs[direction][m].append(tuple([index, belief]))
        return beliefs


    def process_beliefs(self, beliefs_dict, i):
        """ Processes the output of BayesOptExperiment.

        Parameters
        ----------
        beliefs_dict : tuple
            Dictionary of beliefs
        i : int
            Index of loop
        methods : list of str
            The method used to estimate model beliefs

        Returns
        -------
        ts_list : dict
            Long-style belief records
        """
        # instantiate belief list
        belief_list = []

        # record results
        for belief_name, step_beliefs in beliefs_dict.items():

            for step, beliefs in enumerate(step_beliefs):

                # beliefs are a tuple of the element index and the belief value
                belief_index, belief = beliefs

                for bidx in range(len(belief)):
                    
                    belief_list.append(
                        {'Acquisition Function': self.acquisition,
                         'Belief Function': belief_name,
                         'Trial': i,
                         'Round': step,
                         'Index': belief_index[bidx].item(),
                         'Value': belief[bidx].item()}
                    )

        return belief_list


    def process_results(self, acquisitions, ds, i):
        """ Processes the output of BayesOptExperiment.

        Parameters
        ----------
        seen : list of int
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
        # unpack data
        gs, ys = ds

        # for acquisitions each round
        for round_, acquisition in acquisitions.items():
            
            # unpack acquisition indices
            seen, _ = acquisition
            
            # get actual values
            output = ys[seen]
            
            # find the max of the points seen so far
            result = torch.max(output).item()
            self.results.append(
                {
                    'Acquisition Function': self.acquisition,
                    'Trial': i,
                    'Datapoints Acquired': len(seen),
                    'Datapoint Indices': seen,
                    'Round': round_,
                    'Value': result
                }
            )

        return self.results


    def generate_data(self):
        """ Generate data, put on GPU if possible.
        """
        # Load data
        data_func = get_attr(pinot.data, self.data)
        if self.sample_frac != 1.0:
            ds = data_func(sample_frac=self.sample_frac)
        else:
            ds = data_func()

        # Limit to first two fields of tuple
        ds_data = [(d[0], d[1]) for d in ds]
        
        # Gather dates if available
        if len(ds[0]) > 2:
            ds_dates = [d[2] for d in ds]
        else:
            ds_dates = []

        # Batch data
        ds_data = pinot.data.utils.batch(ds_data, len(ds), seed=None)
        
        # Move data to GPU
        ds_data = [tuple([i.to(self.device) for i in ds_data[0]])]
        
        return ds_data[0], ds_dates


    def gather_acquisitions(self, ds_dates):
        """ Annotates acquisition times for human.
        """
        # Bin dates
        _, inverse = np.unique(ds_dates, return_inverse=True)
        
        # construct acquisitions
        acquisitions = OrderedDict()
        ds_range = np.arange(len(ds_dates))
        num_rounds = inverse[-1] + 1

        # for each round
        for round_ in range(num_rounds):
            
            # find mask for the round
            in_round = inverse == round_
            
            # get the first index *after* the mask
            unseen_idx = np.argwhere(in_round)[-1][0] + 1
            
            # use that index to subset before and after
            seen = ds_range[:unseen_idx].tolist()
            unseen = ds_range[unseen_idx:].tolist()
            
            # make acquisition tuple and add to acquisitions
            acquisitions[round_] = tuple([seen, unseen])

        return acquisitions


    def get_acquisition_fn(self):
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
        representation = pinot.representation.SequentialMix(config=self.architecture)

        if hasattr(pinot.regressors, self.net):
            output_regressor = getattr(pinot.regressors, self.net)
            net = pinot.Net(
                representation=representation,
                output_regressor_class=output_regressor,
                n_inducing_points=self.n_inducing_points,
            )

        if self.net == 'semi':
            output_regressor = pinot.regressors.NeuralNetworkRegressor
            net = SemiSupervisedNet(
                representation=representation,
                output_regressor=output_regressor,
                n_inducing_points=self.n_inducing_points,
            )

        elif self.net == 'semi_gp':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor

            net = SemiSupervisedNet(
                representation=representation,
                output_regressor=output_regressor,
                n_inducing_points=self.n_inducing_points,
            )

        elif self.net == 'multitask':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
            net = pinot.multitask.MultitaskNet(
                representation=representation,
                output_regressor=output_regressor,
                n_inducing_points=self.n_inducing_points,
            )

        return net


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='ExactGaussianProcessRegressor')
    parser.add_argument('--architecture', nargs='+', type=str,
        default=['GraphConv', '32', 'activation', 'tanh',
                 'GraphConv', '32', 'activation', 'tanh',
                 'GraphConv', '32', 'activation', 'tanh'])
    parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--output', type=str, default="out", help="Name of folder to store results")


    parser.add_argument('--data', type=str, default='esol')
    parser.add_argument('--sample_frac', type=float, default=0.1, help="Proportion of dataset to read in"
    )
    parser.add_argument('--acquisition', type=str, default='ExpectedImprovement')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--q', type=int, default=1)
    parser.add_argument('--early_stopping', type=int, default=0)
    
    parser.add_argument('--beliefs', nargs='+', type=str,
        default=['ThompsonSampling', 'MaxProbabilityOfImprovement', 'MaxUCB', 'MaxExpectedImprovement']
    )
    parser.add_argument('--num_thompson_samples', type=int, default=500)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--index_provided', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument(
        '--annealing',
        type=float,
        default=1.0,
        help="Scaling factor on the KL term in the variational inference loss"
    )
    parser.add_argument(
        '--n_inducing_points',
        type=int,
        default=100,
        help="Number of inducing points to use for variational inference"
    )


    args = parser.parse_args()

    plot = BeliefActivePlot(
        # net config
        net=args.net,
        architecture=args.architecture,
        n_inducing_points=args.n_inducing_points,
        annealing=args.annealing,

        # optimizer config
        optimizer_type=args.optimizer,
        lr=args.lr,

        # experiment config
        data=args.data,
        sample_frac=args.sample_frac,
        acquisition=args.acquisition,
        num_samples=args.num_samples,
        num_thompson_samples=args.num_thompson_samples,        
        q=args.q,
        early_stopping=args.early_stopping,

        # beliefs
        beliefs=args.beliefs,

        # housekeeping
        device=args.device,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
    )

    # run experiment
    best_df, pro_ts_df, retro_ts_df = plot.generate()

    # write to disk
    beliefs_string = '_'.join(args.beliefs)
    filename = f'{args.net}_{args.optimizer}_{args.data}_num_epochs{args.num_epochs}_{args.acquisition}_q{args.q}_beliefs{beliefs_string}_{args.index}.csv'
    
    best_df.to_csv(f'{args.output}/best_{filename}')
    pro_ts_df.to_csv(f'{args.output}/pro_{filename}')
    retro_ts_df.to_csv(f'{args.output}/retro_{filename}')