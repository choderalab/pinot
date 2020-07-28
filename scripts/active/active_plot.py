import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import pinot
import pinot.active.experiment as experiment
from pinot.generative import SemiSupervisedNet
import os

######################
# Utilities

def _get_dataframe(value_dict):
    """ Creates pandas dataframe to play nice with seaborn
    """
    best_df = pd.DataFrame.from_records(
        [
            (acq_fn, trial, step, value)
            for acq_fn, trial_dict in dict(value_dict).items()
            for trial, value_history in trial_dict.items()
            for step, value in enumerate(value_history)
        ],
        columns = ['Acquisition Function', 'Trial',
                   'Datapoints Acquired', 'Best Solubility']
    )
    return best_df


######################
# Function definitions

class ActivePlot():

    def __init__(self, net, config,
                 lr, optimizer_type,
                 data, unlabeled_data, unlabeled_volume, acquisition, num_samples, q,
                 device, num_trials, num_rounds, num_epochs):

        # net config
        self.net = net
        self.config = config

        # optimizer config
        self.lr = lr
        self.optimizer_type = optimizer_type

        # experiment config
        self.data = data
        self.unlabeled_data = unlabeled_data
        self.unlabeled_volume = unlabeled_volume
        self.acquisition = acquisition
        self.num_samples = num_samples
        self.q = q
        self.train = pinot.app.experiment.Train

        # handle semi
        if self.net == 'semi' or self.net == "semi_gp":
            self.bo_cls = experiment.SemiSupervisedBayesOptExperiment
        elif self.net == 'multitask':
            self.bo_cls = pinot.multitask.MultitaskBayesOptExperiment
            self.train = pinot.multitask.MultitaskTrain
        else:
            self.bo_cls = experiment.BayesOptExperiment

        # housekeeping
        self.device = torch.device(device)
        self.num_trials = num_trials
        self.num_rounds = num_rounds
        self.num_epochs = num_epochs

        # instantiate results dictionary
        self.results = defaultdict(dict)


    def generate(self):
        """
        Performs experiment loops.
        """
        ds = self.generate_data()

        # get results for each trial
        final_results = self.run_trials(ds)

        # create pandas dataframe to play nice with seaborn
        best_df = _get_dataframe(final_results)
        
        return best_df


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

        if self.unlabeled_data:
            unlabeled_data = getattr(pinot.data, self.unlabeled_data)()
            # Use a portion of the unlabeled data
            unlabeled_data, _ = pinot.data.utils.split(unlabeled_data, [self.unlabeled_volume, 1-self.unlabeled_volume])
            unlabeled_data = [(g.to(self.device),y.to(self.device)) for (g,y) in unlabeled_data]

        # acquistion functions to be tested
        for i in range(self.num_trials):

            # make fresh net and optimizer
            net = self.get_net().to(self.device)

            optimizer = pinot.app.utils.optimizer_translation(
                opt_string=self.optimizer_type,
                lr=self.lr,
                weight_decay=0.01,
                kl_loss_scaling=1.0/float(len(ds[0][1]))
                )(net)

            # instantiate experiment
            self.bo = self.bo_cls(
                net=net,
                data=ds[0],
                optimizer=optimizer,
                acquisition=acq_fn,
                num_epochs=self.num_epochs,
                q=self.q,
                slice_fn=experiment._slice_fn_tuple, # pinot.active.
                collate_fn=experiment._collate_fn_graph, # pinot.active.
                train_class=self.train
            )

            if self.unlabeled_data:
                self.bo.unlabeled_data = unlabeled_data

            # run experiment
            x = self.bo.run(num_rounds=self.num_rounds)
            self.results = self.process_results(x, ds, i)

        return self.results


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
        results_size = self.num_rounds * self.q + 1

        if self.net == 'multitask':
            results_data = actual_sol*np.ones((results_size, ys.size(1)))
            output = ys[x]
            output[torch.isnan(output)] = -np.inf
        else:
            results_data = actual_sol*np.ones(results_size)
            output = ys[x]

        results_data[:len(x)] = np.maximum.accumulate(output.cpu().squeeze())

        # record results
        self.results[self.acquisition][i] = results_data
        return self.results


    def generate_data(self):
        """
        Generate data, put on GPU if possible.
        """
        # Load and batch data
        ds = getattr(pinot.data, self.data)()
        if len(ds[0]) > 2: # Temporal Dataset might have 3 fields per tuple
            ds = [(d[0],d[1]) for d in ds]
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
                output_regressor_class=output_regressor
            )

        elif self.net == 'semi_gp':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor

            net = SemiSupervisedNet(
                representation=representation,
                output_regressor_class=output_regressor,
            )

        elif self.net == 'multitask':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
            net = pinot.multitask.MultitaskNet(
                representation=representation,
                output_regressor_class=output_regressor,
            )

        return net


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='ExactGaussianProcessRegressor')
    parser.add_argument('--config', nargs="+", type=str,
        default=["GraphConv", "64", "activation", "tanh",
        "GraphConv", "64", "activation", "tanh",
        "GraphConv", "64", "activation", "tanh",
        "GraphConv", "64", "activation", "tanh"])

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')

    parser.add_argument('--data', type=str, default='esol')
    parser.add_argument('--unlabeled_data', type=str, default="")
    parser.add_argument('--acquisition', type=str, default='ExpectedImprovement')
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--q', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=100)
    parser.add_argument('--num_epochs', type=int, default=200)

    parser.add_argument('--index_provided', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output_folder', type=str, default="plotting_active")
    parser.add_argument('--seed', type=int, default=2666, help="Seed for weight initialization")
    parser.add_argument('--unlabeled_volume', type=float, default=0.2, help="volume of unlabeled data to use")
    
    args = parser.parse_args()
    print(args)
    
    torch.manual_seed(args.seed) # Pick a seed

    plot = ActivePlot(
        # net config
        net=args.net,
        config=args.config,

        # optimizer config
        optimizer_type=args.optimizer,
        lr=args.lr,

        # experiment config
        data=args.data,
        unlabeled_data=args.unlabeled_data,
        unlabeled_volume=args.unlabeled_volume,
        acquisition=args.acquisition,
        num_samples=args.num_samples,
        q=args.q,

        # housekeeping
        device=args.device,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
    )

    best_df = plot.generate()
    representation = args.config[0]

    # If the output folder doesn't exist, create one
    if not os.path.isdir(args.output_folder):
        os.mkdir(args.output_folder)

    # save to disk
    if args.index_provided:
        filename =\
        f'{args.net}_{args.data}_{args.acquisition}_q{args.q}_{args.index}_{args.num_epochs}_{args.seed}_{args.unlabeled_volume}.csv'
    best_df.to_csv(os.path.join(args.output_folder, filename))
