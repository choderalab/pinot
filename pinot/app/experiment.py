# =============================================================================
# IMPORTS
# =============================================================================
import torch
import dgl
import argparse
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import math
from datetime import datetime
import os
from abc import ABC
import copy
import logging
import pickle
import pinot
from datetime import datetime, timedelta

# =============================================================================
# MODULE FUNCTIONS - *STATELESS*
# =============================================================================

def train(
    net,
    data,
    optimizer,
    n_epochs=100,
    record_interval=1,
    lr_scheduler=None,
    annealing=1.0,
    logging=None,
    state_save_file=None,
    time_limit=None,
    out_dir='out'
    ):
    """
    Train the model for multiple steps
    and record the weights once every `record_interval`.

    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    Methods
    -------
    train_once : Train model once.

    train : Train the model multiple times.

    """
    def train_once(net, data, optimizer, annealing):
        """
        TODO: FIX THE LOGGING MECHANISM
        Train the model for one batch.
        """
        total_loss = 0.
        for d in data:
            
            batch_ratio = len(d[1]) / len(data)

            def l():
                """ """
                optimizer.zero_grad()
                loss = torch.sum(
                    net.loss(
                        *d,
                        kl_loss_scaling=batch_ratio,
                        annealing=annealing
                    )
                )
                loss.backward()
                # not sure how to log in a purely functional way
                # loss_temp = loss.detach().cpu()
                return loss

            optimizer.step(l)
            # total_loss += self.loss_temp / len(d[1])
        # mean_loss = total_loss / len(self.data)
        # return mean_loss

    # set time limit
    start_time = datetime.now()
    if time_limit:
        hours, minutes = (int(t) for t in time_limit.split(':'))
        limit_delta = timedelta(hours=hours, minutes=minutes)
    else:
        limit_delta = timedelta(days=365)

    states = {}
    for epoch_idx in range(int(n_epochs)):
        
        mean_loss = train_once(net, data, optimizer, annealing)

        if lr_scheduler:
            lr_scheduler.step()

        if epoch_idx % record_interval == 0:
            
            states[epoch_idx] = copy.deepcopy(net.state_dict())

            if logging:
                logging.debug(f'Epoch {epoch_idx} average loss: {mean_loss}')

            # TODO: fix this hack
            if state_save_file:
                pickle.dump(
                    states,
                    open(f'{out_dir}/dict_state_{state_save_file}.p', 'wb')
                )

        # check if we've hit our time limit
        if (datetime.now() - start_time) > limit_delta:
            break

    states[f'final_{epoch_idx}'] = copy.deepcopy(net.state_dict())

    if hasattr(optimizer, "expectation_params"):
        optimizer.expectation_params()

    return states


def test(
    net,
    data,
    states,
    sampler=None,
    metrics=[pinot.rmse, pinot.r2]
    ):
    """ Test experiment. Metrics are applied to the saved states of the
    model to characterize its performance.


    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        Metrics used to characterize the performance.

    Methods
    -------
    test : Run the test experiment.

    """
    def compute_conditional(net, data, batch_size):
        # compute conditional distribution in batched fashion
        locs, scales = [], []
        for idx, d in enumerate(data.batch(batch_size, partial_batch=True)):

            g_batch, _ = d
            distribution_batch = net.condition(g_batch)
            loc_batch = distribution_batch.mean.flatten().cpu()
            scale_batch = distribution_batch.variance.pow(0.5).flatten().cpu()
            locs.append(loc_batch)
            scales.append(scale_batch)

        distribution = torch.distributions.normal.Normal(
            loc=torch.cat(locs),
            scale=torch.cat(scales)
        )
        return distribution

    # switch to test
    net.eval()

    # initialize an empty dict for each metrics
    results = {}

    for metric in metrics:
        results[metric.__name__] = {}

    # make g, y into single batches
    g, y = data.batch(len(data))[0]
    for state_name, state in states.items():  # loop through states
        
        net.load_state_dict(state)

        if net.has_exact_gp:
            batch_size = len(data)
        else:
            batch_size = 32
        
        # compute conditional distribution in batched fashion
        distribution = compute_conditional(net, data, batch_size)
        y = y.detach().cpu().reshape(-1, 1)
        for metric in metrics:  # loop through the metrics
            results[metric.__name__][state_name] = (
                metric(
                    net,
                    distribution,
                    y,
                    sampler=sampler,
                    batch_size=batch_size
                )
                .detach()
                .cpu()
                .numpy()
            )

    return results


def train_and_test(
    net,
    data_tr,
    data_te,
    optimizer,
    metrics=[pinot.rmse, pinot.r2, pinot.pearsonr, pinot.avg_nll, pinot.absolute_error],
    n_epochs=100,
    record_interval=1,
    lr_scheduler=None,
    annealing=1.0,
    logging=None,
    state_save_file=None,
    time_limit=None,
    out_dir='out'
    ):
    """ Run training and test experiment.

    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        (Default value: `[pinot.rmse, pinot.r2, pinot.pearsonr, pinot.avg_nll]`)
        Metrics used to characterize the performance.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    train_cls: `Train`
        (Default value = Train)
        Train class to use

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    logging: 
        (Default value = None)
        A preconfigured logging object that can send to disk the average epoch loss

    state_save_file : str
        (Default value = None)
    """
    print('training now')
    
    states = train(
        net=net,
        data=data_tr,
        optimizer=optimizer,
        n_epochs=n_epochs,
        lr_scheduler=lr_scheduler,
        annealing=annealing,
        logging=logging,
        state_save_file=state_save_file,
        time_limit=time_limit,
        out_dir=out_dir
    )

    print('testing now')
    
    results_tr = test(
        net=net,
        data=data_tr,
        metrics=metrics,
        states=states,
        sampler=optimizer,
    )

    results_te = test(
        net=net,
        data=data_te,
        metrics=metrics,
        states=states,
        sampler=optimizer,
    )

    return {"train": results_tr, "test": results_te}




# =============================================================================
# MODULE CLASSES
# =============================================================================


class Train:
    """Training experiment.

    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    Methods
    -------
    train_once : Train model once.

    train : Train the model multiple times.




    """

    def __init__(
        self,
        net,
        data,
        optimizer,
        n_epochs=100,
        record_interval=1,
        lr_scheduler=None,
        annealing=1.0,
        logging=None,
        state_save_file=None
    ):

        self.data = data
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.net = net
        self.record_interval = record_interval
        self.states = {}
        self.lr_scheduler = lr_scheduler
        self.annealing = annealing
        self.logging = logging
        self.state_save_file = state_save_file

    def train_once(self):
        """
        TODO: FIX THE LOGGING MECHANISM
        Train the model for one batch.
        """
        total_loss = 0.
        for d in self.data:
            
            batch_ratio = len(d[1]) / len(self.data)

            def l():
                """ """
                self.optimizer.zero_grad()
                loss = torch.sum(
                    self.net.loss(
                        *d,
                        kl_loss_scaling=batch_ratio,
                        annealing=self.annealing
                    )
                )
                loss.backward()
                self.loss_temp = loss.detach().cpu()
                return loss

            self.optimizer.step(l)
            total_loss += self.loss_temp / len(d[1])

        mean_loss = total_loss / len(self.data)
        return mean_loss

    def train(self):
        """Train the model for multiple steps and
        record the weights once every
        `record_interval`.

        Parameters
        ----------

        Returns
        -------

        """

        for epoch_idx in range(int(self.n_epochs)):
            
            mean_loss = self.train_once()

            if self.lr_scheduler:
                self.lr_scheduler.step()

            if epoch_idx % self.record_interval == 0:
                
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

                if logging:
                    logging.debug(f'Epoch {epoch_idx} average loss: {mean_loss}')

                # TODO: fix this hack
                if self.state_save_file:
                    pickle.dump(
                        self.states,
                        open(f'./out/dict_state_{self.state_save_file}.p', 'wb')
                    )


        self.states["final"] = copy.deepcopy(self.net.state_dict())

        if hasattr(self.optimizer, "expectation_params"):
            self.optimizer.expectation_params()

        return self


class Test:
    """ Test experiment. Metrics are applied to the saved states of the
    model to characterize its performance.


    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        Metrics used to characterize the performance.

    Methods
    -------
    test : Run the test experiment.

    """

    def __init__(
        self, net, data, states, sampler=None, metrics=[pinot.rmse, pinot.r2]
    ):
        self.net = net  # deepcopy the model object
        self.data = data
        self.metrics = metrics
        self.states = states
        self.sampler = sampler

    def test(self):
        """ Run test experiment. """
        
        def _independent(distribution):
            """ Make predictive distribution for test set independent.

            Parameters
            ----------
            distribution : `torch.distribution.Distribution`
                Input distribution.

            Returns
            -------
            distribution : `torch.distribution.Distribution`
                Output distribution.

            """
            return torch.distributions.normal.Normal(
                loc=distribution.mean.flatten(),
                scale=distribution.variance.pow(0.5).flatten(),
        )

        # switch to test
        self.net.eval()

        # initialize an empty dict for each metrics
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = {}

        # make g, y into single batches
        g, y = self.data.batch(len(self.data))[0]
        for state_name, state in self.states.items():  # loop through states
            
            self.net.load_state_dict(state)

            if self.net.has_exact_gp:
                batch_size = len(self.data)
            else:
                batch_size = 32

            # compute conditional distribution in batched fashion
            locs, scales = [], []
            for idx, d in enumerate(self.data.batch(batch_size, partial_batch=True)):
    
                g_batch, _ = d
                distribution_batch = self.net.condition(g_batch)
                loc_batch = distribution_batch.mean.flatten().cpu()
                scale_batch = distribution_batch.variance.pow(0.5).flatten().cpu()
                locs.append(loc_batch)
                scales.append(scale_batch)

            distribution = torch.distributions.normal.Normal(
                loc=torch.cat(locs),
                scale=torch.cat(scales)
            )

            y = y.detach().cpu().reshape(-1, 1)
            for metric in self.metrics:  # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric(
                        self.net,
                        distribution,
                        y,
                        sampler=self.sampler,
                        batch_size=batch_size
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )


        self.results = results
        return self


class TrainAndTest:
    """ Run training and test experiment.

    Parameters
    ----------
    net : `pinot.Net`
        Forward pass model that combines representation and output regression
        and generates predictive distribution.

    data : `List` of `tuple` of `(dgl.DGLGraph, torch.Tensor)`
        or `pinot.data.dataset.Dataset`
        Pairs of graph, measurement.

    optimizer : `torch.optim.Optimizer` or `pinot.Sampler`
        Optimizer for training.

    metrics : `List` of `callable`
        (Default value: `[pinot.rmse, pinot.r2, pinot.pearsonr, pinot.avg_nll]`)
        Metrics used to characterize the performance.

    n_epochs : `int`
        Number of epochs.

    record_interval : `int`
        (Default value = 1)
        Interval states are recorded.

    train_cls: `Train`
        (Default value = Train)
        Train class to use

    lr_scheduler: `torch.optim.lr_scheduler`
        (Default value = None)
        Learning rate scheduler, will apply after every training epoch

    logging: 
        (Default value = None)
        A preconfigured logging object that can send to disk the average epoch loss

    state_save_file : str
        (Default value = None)

    Methods
    -------
    run : conduct experiment

    """

    def __init__(
        self,
        net,
        data_tr,
        data_te,
        optimizer,
        metrics=[pinot.rmse, pinot.r2, pinot.pearsonr, pinot.avg_nll],
        n_epochs=100,
        record_interval=1,
        train_cls=Train,
        lr_scheduler=None,
        annealing=1.0,
        logging=None,
        state_save_file=None
    ):
        self.net = net  # deepcopy the model object
        self.data_tr = data_tr
        self.data_te = data_te
        self.metrics = metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.record_interval = record_interval
        self.train_cls = train_cls
        self.lr_scheduler = lr_scheduler
        self.annealing = annealing
        self.logging = logging
        self.state_save_file = state_save_file

    def __str__(self):
        _str = ""
        _str += "# model"
        _str += "\n"
        _str += str(self.net)
        _str += "\n"
        if hasattr(self.net, "noise_model"):
            _str += "# noise model"
            _str += "\n"
            _str += str(self.net.noise_model)
            _str += "\n"
        _str += "# optimizer"
        _str += "\n"
        _str += str(self.optimizer)
        _str += "\n"
        _str += "# n_epochs"
        _str += "\n"
        _str += str(self.n_epochs)
        _str += "\n"
        return _str

    def run(self):
        """ Run train and test experiments. """
        
        print('training now')
        
        tr = self.train_cls(
            net=self.net,
            data=self.data_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            lr_scheduler=self.lr_scheduler,
            annealing=self.annealing,
            logging=self.logging,
            state_save_file=self.state_save_file
        ).train()

        self.states = tr.states
        del tr

        print('testing now')
        
        te_data_tr = Test(
            net=self.net,
            data=self.data_tr,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        ).test()

        self.results_tr = te_data_tr.results

        del te_data_tr

        te_data_te = Test(
            net=self.net,
            data=self.data_te,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        ).test()

        self.results_te = te_data_te.results

        del te_data_te

        return {"test": self.results_te, "training": self.results_tr}


class MultipleTrainAndTest:
    """A sequence of controlled experiment."""

    def __init__(self, experiment_generating_fn, param_dicts):
        self.experiment_generating_fn = experiment_generating_fn
        self.param_dicts = param_dicts

    def run(self):
        """ """
        results = []

        for param_dict in self.param_dicts:
            train_and_test = self.experiment_generating_fn(param_dict)
            result = train_and_test.run()
            results.append((param_dict, result))
            del train_and_test

        self.results = results

        return self.results
