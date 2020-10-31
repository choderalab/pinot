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
import pinot

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
    ):

        self.data = data
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.net = net
        self.record_interval = record_interval
        self.states = {}
        self.lr_scheduler = lr_scheduler

    def train_once(self):
        """Train the model for one batch."""
        for d in self.data:

            def l():
                """ """
                self.optimizer.zero_grad()
                loss = torch.sum(self.net.loss(*d))
                loss.backward()
                return loss

            self.optimizer.step(l)

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
            self.train_once()
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        self.states["final"] = copy.deepcopy(self.net.state_dict())

        if hasattr(self.optimizer, "expecation_params"):
            self.optimizer.expectation_params()

        return self.net


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
        # switch to test
        self.net.eval()

        # initialize an empty dict for each metrics
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = {}

        for state_name, state in self.states.items():  # loop through states
            self.net.load_state_dict(state)

            # concat y and y_hat in test set

            # y = []
            # g = []
            # for g_, y_ in self.data:
            #     y.append(y_)
            #     if isinstance(g_, dgl.DGLGraph):
            #         g.append(g_)
            #     else:
            #         g += dgl.unbatch(g_)
            #
            # if y[0].dim() == 0:
            #     y = torch.stack(y)
            # else:
            #     y = torch.cat(y)
            #
            # g = dgl.batch(g)
            def _batch(x):
                if isinstance(x[0], dgl.DGLGraph):
                    _x = []
                    for _g in x:
                        _x += dgl.unbatch(_g)
                    return dgl.batch(_x)

                else:
                    if x[0].dim() == 0:
                        return torch.stack(x)

                    else:
                        return torch.cat(x)

            xs = list(zip(*self.data))
            xs = [_batch(x) for x in xs]

            # get input and auxiliary arguments
            g = xs[0]
            y = xs[1]
            _args = xs[2:]

            for metric in self.metrics:  # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric(
                        self.net,
                        g,
                        y,
                        *_args,
                        sampler=self.sampler,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )


        self.results = results
        return dict(results)


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
        train = self.train_cls(
            net=self.net,
            data=self.data_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
            lr_scheduler=self.lr_scheduler,
        )

        print('training now')

        train.train()

        self.states = train.states

        print('testing now')
        test = Test(
            net=self.net,
            data=self.data_te,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        )

        test.test()

        self.results_te = test.results

        test = Test(
            net=self.net,
            data=self.data_tr,
            metrics=self.metrics,
            states=self.states,
            sampler=self.optimizer,
        )

        test.test()

        self.results_tr = test.results

        del train

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
