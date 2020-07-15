import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd

import torch
import pinot
import pinot.active.experiment as experiment
from pinot.generative import SemiSupervisedNet


######################
# Function definitions

class ActivePlot():

    def __init__(self, net, config,
                 lr, optimizer_type,
                 data, acquisition, num_samples, q,
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
        best_df = pd.DataFrame.from_records(
            [
                (acq_fn, trial, step, best)
                for acq_fn, trial_dict in dict(final_results).items()
                for trial, best_history in trial_dict.items()
                for step, best in enumerate(best_history)
            ],
            columns=['Acquisition Function', 'Trial', 'Datapoints Acquired', 'Best Solubility']
        )

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
        gs, ys = ds[0]
        self.feat_dim = gs.ndata['h'].shape[1]
        actual_sol = torch.max(ys).item()

        acq_fn = self.get_acquisition(gs)

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
            self.bo = self.bo_cls(
                net=net,
                data=ds[0],
                optimizer=optimizer(net),
                strategy=self.strategy,
                acquisition=acq_fn,
                n_epochs=self.num_epochs,
                q=self.q,
                slice_fn=experiment._slice_fn_tuple, # pinot.active.
                collate_fn=experiment._collate_fn_graph, # pinot.active.
                train_class=self.train
            )

            # run experiment
            x = self.bo.run(num_rounds=self.num_rounds)

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
        ds = pinot.data.utils.batch(ds, len(ds), seed=None)
        ds = [tuple([i.to(self.device) for i in ds[0]])]
        return ds

    def get_acquisition(self, gs):
        """ Retrieve acquisition function and prepare for BO Experiment
        """
        sequential_acquisitions = {
            'ExpectedImprovement': pinot.active.acquisition.expected_improvement_analytical,
            'ProbabilityOfImprovement': pinot.active.acquisition.probability_of_improvement,
            'UpperConfidenceBound': pinot.active.acquisition.upper_confidence_bound,
            'Uncertainty': pinot.active.acquisition.uncertainty,
            'Human': pinot.active.acquisition.temporal,
            'Random': pinot.active.acquisition.random,
            
        }

        batch_acquisitions = {
            'ThompsonSampling': pinot.active.batch_acquisition.thompson_sampling,
            'WeightedSamplingExpectedImprovement': pinot.active.batch_acquisition.exponential_weighted_ei_analytical,
            'WeightedSamplingProbabilityOfImprovement': pinot.active.batch_acquisition.exponential_weighted_pi,
            'WeightedSamplingUpperConfidenceBound': pinot.active.batch_acquisition.exponential_weighted_ucb,
            'GreedyExpectedImprovement': pinot.active.batch_acquisition.greedy_ei_analytical,
            'GreedyProbabilityOfImprovement': pinot.active.batch_acquisition.greedy_pi,
            'GreedyUpperConfidenceBound': pinot.active.batch_acquisition.greedy_ucb,
            'BatchRandom': pinot.active.batch_acquisition.batch_random,
            'BatchTemporal': pinot.active.batch_acquisition.batch_temporal
        }

        if self.acquisition in sequential_acquisitions:
            self.strategy = 'sequential'
            acq_fn = sequential_acquisitions[self.acquisition]
        
        elif self.acquisition in batch_acquisitions:
            self.strategy = 'batch'
            acq_fn = batch_acquisitions[self.acquisition]

        return acq_fn


    def get_net(self):
        """
        Retrive GP using representation provided in args.
        """
        representation = pinot.representation.SequentialMix(
            config=self.config)

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
    parser.add_argument('--q', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=10)

    parser.add_argument('--index_provided', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    plot = ActivePlot(
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