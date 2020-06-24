import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import pinot

import sys
sys.path.append('../../pinot/active/')

import experiment

sys.path.append('../../pinot/')
from multitask import MultitaskNet
from multitask.experiment import MultitaskTrain
from pinot.generative import SemiSupervisedNet


######################
# Function definitions

class ActivePlot():

    def __init__(self, net, layer, config,
                 lr, optimizer_type, weighted_acquire,
                 data, strategy, acquisition, marginalize_batch, num_samples, q,
                 device, num_trials, num_rounds, num_epochs):        
        
        # net config
        self.net = net
        self.layer = layer
        self.config = config

        # optimizer config
        self.lr = lr
        self.optimizer_type = optimizer_type
        
        # experiment config
        self.data = data
        self.strategy = strategy
        self.acquisition = acquisition
        self.marginalize_batch = marginalize_batch
        self.weighted_acquire = weighted_acquire
        self.num_samples = num_samples
        self.q = q
        self.train = pinot.app.experiment.Train

        # handle semi
        if self.net == 'semi':
            self.bo = experiment.SemiSupervisedBayesOptExperiment
        elif self.net == 'multitask':
            self.bo = experiment.MultitaskBayesOptExperiment
            self.train = MultitaskTrain
        else:
            self.bo = experiment.BayesOptExperiment

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
            bo = self.bo(net=net,
                         data=ds[0],
                         optimizer=optimizer(net),
                         acquisition=acq_fn,
                         n_epochs=self.num_epochs,
                         strategy=self.strategy,
                         q=self.q,
                         weighted_acquire=self.weighted_acquire,
                         slice_fn=experiment._slice_fn_tuple, # pinot.active.
                         collate_fn=experiment._collate_fn_graph, # pinot.active.
                         train_class=self.train
                         )

            # run experiment
            x = bo.run(num_rounds=self.num_rounds)

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
        
        '''        
        batch_acquisitions = {'Expected Improvement': SeqAcquire(acq_fn='ei'),
                              'Probability of Improvement': SeqAcquire(acq_fn='pi'),
                              'Upper Confidence Bound': SeqAcquire(acq_fn='ucb', beta=0.95),
                              'Uncertainty': SeqAcquire(acq_fn='uncertainty'),
                              'Random': SeqAcquire(acq_fn='random')}
        '''
        sequential_acquisitions = {'ExpectedImprovement': pinot.active.acquisition.expected_improvement,
                                   'ProbabilityOfImprovement': pinot.active.acquisition.probability_of_improvement,
                                   'UpperConfidenceBound': pinot.active.acquisition.expected_improvement,
                                   'Uncertainty': pinot.active.acquisition.expected_improvement,
                                   'Human': pinot.active.acquisition.dummy,
                                   'Random': pinot.active.acquisition.expected_improvement}
        
        if self.strategy == 'batch':
            acq_fn = batch_acquisitions[self.acquisition]
            acq_fn = MCAcquire(sequential_acq=acq_fn, batch_size=gs.batch_size,
                               q=self.q,
                               marginalize_batch=self.marginalize_batch,
                               num_samples=self.num_samples)
        else:
            acq_fn = sequential_acquisitions[self.acquisition]

        return acq_fn


    def get_net(self):
        """
        Retrive GP using representation provided in args.
        """
        layer = pinot.representation.dgl_legacy.gn(model_name=self.layer)

        representation = pinot.representation.Sequential(
            layer=layer,
            config=self.config)

        if self.net == 'gp':
            kernel = pinot.inference.gp.kernels.deep_kernel.DeepKernel(
                    representation=representation,
                    base_kernel=pinot.inference.gp.kernels.rbf.RBF())
            net = pinot.inference.gp.gpr.exact_gpr.ExactGPR(kernel)

        elif self.net == 'mle':
            net = pinot.Net(representation)

        elif self.net == 'semi':
            net = SemiSupervisedNet(
                representation=representation,
                unsup_scale=0.1 # <------ if unsup_scale = 0., reduces to supervised model
            )

        elif self.net == 'semi_gp':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor

            net = SemiSupervisedNet(
                representation=representation,
                output_regressor=output_regressor,
            )

        elif self.net == 'multitask':
            output_regressor = pinot.regressors.ExactGaussianProcessRegressor
            net = MultitaskNet(
                representation=representation,
                output_regressor=output_regressor,
            )

        if self.strategy == 'batch':
            net = BTModel(net)

        return net


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='ExactGaussianProcessRegressor')
    parser.add_argument('--representation', type=str, default='GraphConv')

    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--data', type=str, default='esol')
    parser.add_argument('--strategy', type=str, default='sequential')
    parser.add_argument('--acquisition', type=str, default='ExpectedImprovement')
    parser.add_argument('--marginalize_batch', type=bool, default=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--weighted_acquire', type=bool, default=True)
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
        layer=args.representation,
        config=[32, 'tanh', 32, 'tanh', 32, 'tanh'],
        
        # optimizer config
        optimizer_type=args.optimizer,
        lr=args.lr,

        # experiment config
        data=args.data,
        strategy=args.strategy,
        acquisition=args.acquisition,
        marginalize_batch=args.marginalize_batch,
        num_samples=args.num_samples,
        weighted_acquire=args.weighted_acquire,
        q=args.q,

        # housekeeping
        device=args.device,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        )

    best_df = plot.generate()

    # save to disk
    if args.index_provided and args.weighted_acquire:
        filename = f'{args.net}_{args.representation}_{args.optimizer}_{args.data}_{args.strategy}_{args.acquisition}_q{args.q}_weighted_{args.index}.csv'
    elif args.index_provided and not args.weighted_acquire:
        filename = f'{args.net}_{args.representation}_{args.optimizer}_{args.data}_{args.strategy}_{args.acquisition}_q{args.q}_unweighted_{args.index}.csv'    
    best_df.to_csv(filename)