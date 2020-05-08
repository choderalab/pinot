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
class Train():
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
                loss = torch.sum(self.net.loss(g, y))
                self.optimizer.zero_grad()
                loss.backward()
                return loss
            
            self.optimizer.step(l)
            

    def train(self):
        """ Train the model for multiple steps and
        record the weights once every
        `record_interval`.

        """

        for epoch_idx in range(self.n_epochs):
            self.train_once()
            
            if epoch_idx // self.record_interval == 0:
                self.states[epoch_idx] = self.net.state_dict()

        self.states['final'] = self.net.state_dict()


class Test():
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
            self, 
            net, 
            data, 
            states,
            metrics=[
                pinot.rmse,
                pinot.r2
            ]): 
        self.net = copy.deepcopy(net) # deepcopy the model object
        self.data = data
        self.metrics = metrics
        self.states = states

    def test(self):
        # initialize an empty dict for each metrics
        results = {}

        for metric in self.metrics:
            results[metric.__name__] = {}

        for state_name, state in self.states.items(): # loop through states
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

            for metric in self.metrics: # loop through the metrics
                results[metric.__name__][state_name] = metric(self.net, g, y)
                
        self.results = results
        return self.results

class TrainAndTest():
    """ Train a model and then test it.

    """
    def __init__(
            self, 
            net, 
            data_tr,
            data_te,
            optimizer, 
            metrics=[pinot.rmse, pinot.r2],
            n_epochs=100, 
            record_interval=1):
        self.net = copy.deepcopy(net) # deepcopy the model object
        self.data_tr = data_tr
        self.data_te = data_te
        self.metrics = metrics
        self.optimizer = optimizer
        self.n_epochs = n_epochs
        self.record_interval = record_interval


    def run(self):
        train = Train(
            net=self.net,
            data=self.data_tr,
            optimizer=self.optimizer,
            n_epochs=self.n_epochs)

        self.states = train.states

        test = Test(
            net=self.net,
            data=self.data_te,
            metrics=self.metrics,
            states=self.states)

        self.results_te = test.results

        test = Test(
            net=self.net,
            data=self.data_tr,
            metrics=self.metrics,
            states=self.states)

        self.results_tr = test.results








