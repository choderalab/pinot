import warnings
warnings.filterwarnings("ignore")

from collections import defaultdict

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch

import pinot
from pinot.active.acquisition import BTModel, SeqAcquire, MCAcquire

######################
# Function definitions

class ActivePlot():

    def __init__(self, net, layer, config,
                 lr, optimizer_type,
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
        self.num_samples = num_samples
        self.q = q        

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


    def generate_data(self):
        """
        Generate data, put on GPU if possible.
        """
        # Load and batch data
        ds = getattr(pinot.data, self.data)()
        ds = pinot.data.utils.batch(ds, len(ds), seed=None)
        ds = [tuple([i.to(self.device) for i in ds[0]])]
        return ds


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
        gs, ys = ds[0]
        actual_sol = torch.max(ys).item()
        acq_fn = self.get_acquisition(gs)
        
        # acquistion functions to be tested
        for i in range(self.num_trials):
            
            # make fresh net and optimizer
            net = self.get_net().to(self.device)
            
            optimizer = pinot.app.utils.optimizer_translation(
                opt_string=self.optimizer_type,
                lr=self.lr,
                weight_decay=0.01,
                kl_loss_scaling=1.0/float(len(ds[0][1]))
                )
            
            # instantiate experiment
            bo = pinot.active.experiment.SingleTaskBayesianOptimizationExperiment(
                net=net,
                data=ds[0],
                optimizer=optimizer(net),
                acquisition=acq_fn,
                n_epochs=self.num_epochs,
                strategy=self.strategy,
                q=self.q,
                slice_fn = pinot.active.experiment._slice_fn_tuple,
                collate_fn = pinot.active.experiment._collate_fn_graph
                )

            # run experiment
            x = bo.run(num_rounds=self.num_rounds)

            # pad if experiment stopped early
            # candidates_acquired = limit + 1 because we begin with a blind pick
            results_shape = self.num_rounds * self.q + 1
            results_data = actual_sol*np.ones(results_shape)
            results_data[:len(x)] = np.maximum.accumulate(ys[x].cpu().squeeze())
            
            # print(len(x), results_data[-1])
            # record results
            self.results[self.acquisition][i] = results_data

        return self.results

    def get_acquisition(self, gs):
        """ Retrieve acquisition function and prepare for BO Experiment
        """
        batch_acquisitions = {'Expected Improvement': SeqAcquire(acq_fn='ei'),
                              'Probability of Improvement': SeqAcquire(acq_fn='pi'),
                              'Upper Confidence Bound': SeqAcquire(acq_fn='ucb', beta=0.95),
                              'Uncertainty': SeqAcquire(acq_fn='uncertainty'),
                              'Random': SeqAcquire(acq_fn='random')}

        sequential_acquisitions = {'Expected Improvement': pinot.active.acquisition.expected_improvement,
                                   'Probability of Improvement': pinot.active.acquisition.probability_of_improvement,
                                   'Upper Confidence Bound': pinot.active.acquisition.expected_improvement,
                                   'Uncertainty': pinot.active.acquisition.expected_improvement,
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

        net_representation = pinot.representation.Sequential(
            layer=layer,
            config=self.config)

        if self.net == 'gp':
            kernel = pinot.inference.gp.kernels.deep_kernel.DeepKernel(
                    representation=net_representation,
                    base_kernel=pinot.inference.gp.kernels.rbf.RBF())
            net = pinot.inference.gp.gpr.exact_gpr.ExactGPR(kernel)

        elif self.net == 'mle':
            net = pinot.Net(net_representation)

        if self.strategy == 'batch':
            net = BTModel(net)

        return net


if __name__ == '__main__':

    # Running functions
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--net', type=str, default='gp')
    parser.add_argument('--representation', type=str, default='GraphConv')

    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--optimizer', type=str, default='Adam')
    
    parser.add_argument('--data', type=str, default='esol')
    parser.add_argument('--strategy', type=str, default='sequential')
    parser.add_argument('--acquisition', type=str, default='Expected Improvement')
    parser.add_argument('--marginalize_batch', type=bool, default=True)
    parser.add_argument('--num_samples', type=int, default=1000)
    parser.add_argument('--q', type=int, default=1)

    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--num_trials', type=int, default=1)
    parser.add_argument('--num_rounds', type=int, default=50)
    parser.add_argument('--num_epochs', type=int, default=10)
    
    parser.add_argument('--index_provided', type=bool, default=True)
    parser.add_argument('--index', type=int, default=0)

    args = parser.parse_args()

    Plot = ActivePlot(
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
        q=args.q,

        # housekeeping
        device=args.device,
        num_trials=args.num_trials,
        num_rounds=args.num_rounds,
        num_epochs=args.num_epochs,
        )

    best_df = Plot.generate()

    # save to disk
    if args.index_provided:
        filename = f'{args.net}_{args.representation}_{args.optimizer}_{args.strategy}_q{args.q}_{args.index}.csv'
    best_df.to_csv(filename)