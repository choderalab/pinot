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
import gpytorch

# =============================================================================
# MODULE CLASSES
# =============================================================================


class Train:
    """ Training experiment.

    Attributes
    ----------
    data : generator
        training data
    net : `pinot.Net` object
        model with parameters to be trained
    record_interval: int, default=1
        interval at which `states` are being recorded
    optimizer : `torch.optim.Optimizer` object, default=`torch.optim.Adam(1e-5)`
        optimizer used for training
    n_epochs : int, default=100
        number of epochs


    """

    def __init__(
        self,
        net,
        data,
        optimizer,
        n_epochs=100,
        record_interval=1):

        self.data = data
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.net = net
        self.record_interval = record_interval
        self.states = {}

    def train_once(self):
        """ Train the model for one batch.
        """
        for g, y in self.data:

            def l():
                self.optimizer.zero_grad()
                loss = torch.sum(self.net.loss(g, y))
                loss.backward()
                return loss

            self.optimizer.step(l)

    def train(self):
        """ Train the model for multiple steps and
        record the weights once every
        `record_interval`.

        """

        for epoch_idx in range(int(self.n_epochs)):
            self.train_once()

            if epoch_idx % self.record_interval == 0:
                self.states[epoch_idx] = copy.deepcopy(self.net.state_dict())

        self.states["final"] = copy.deepcopy(self.net.state_dict())

        if hasattr(self.optimizer, "expecation_params"):
            self.optimizer.expectation_params()

        return self.net


class Test:
    """ Run sequences of test on a trained model.

    Attributes
    ----------
    net : `pinot.Net` object
        trained model to be tested
    data : `pinot.data` object
        test data
    metrics : list of `pinot.metrics` functions
        metrics used for testing
    states : dictionary
        nested dictionary of state dicts


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
        # switch to test
        self.net.eval()

        # initialize an empty dict for each metrics
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = {}

        for state_name, state in self.states.items():  # loop through states
            self.net.load_state_dict(state)

            # concat y and y_hat in test set

            y = []
            g = []
            for g_, y_ in self.data:
                y.append(y_)
                g += dgl.unbatch(g_)

            if y[0].dim() == 0:
                y = torch.stack(y)
            else:
                y = torch.cat(y)

            g = dgl.batch(g)

            for metric in self.metrics:  # loop through the metrics
                results[metric.__name__][state_name] = (
                    metric(self.net, g, y, sampler=self.sampler)
                    .detach()
                    .cpu()
                    .numpy()
                )

        self.results = results
        return dict(results)


class TrainAndTest:
    """ Train a model and then test it.

    """

    def __init__(
        self,
        net,
        data_tr,
        data_te,
        optimizer,
        metrics=[pinot.rmse, pinot.r2, pinot.avg_nll],
        n_epochs=100,
        record_interval=1,
    ):
        self.net = net  # deepcopy the model object
        self.data_tr = data_tr
        self.data_te = data_te
        self.metrics = metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.record_interval = record_interval

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
        train = Train(
            net=self.net,
            data=self.data_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs,
        )

        train.train()

        self.states = train.states

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
    """ A sequence of controlled experiment.


    """

    def __init__(self, experiment_generating_fn, param_dicts):
        self.experiment_generating_fn = experiment_generating_fn
        self.param_dicts = param_dicts

    def run(self):
        results = []

        for param_dict in self.param_dicts:
            train_and_test = self.experiment_generating_fn(param_dict)
            result = train_and_test.run()
            results.append((param_dict, result))
            del train_and_test

        self.results = results

        return self.results
