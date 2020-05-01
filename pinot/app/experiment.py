# =============================================================================
# IMPORTS
# =============================================================================
import pinot
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

# =============================================================================
# MODULE CLASSES
# =============================================================================
class Train():
    """ Training experiment.

    Attributes
    ----------
    data : generator
        training data
    optimizer : `torch.optim.Optimizer` object
        optimizer used for training
    n_epochs : int
        number of epochs
    net : `pinot.Net` object
        model with parameters to be trained
    


    """

    def __init__(
            self,
            net,
            data,
            optimizer,
            n_epochs,
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
                opt.zero_grad()
                loss.backward()
                return loss
            self.opt.step(l)

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

    """
    def __init__(self, net, data, metrics, states):
        import copy
        self.net = copy.deepcopy(net) # deepcopy the model object
        self.metrics = metrics
        self.states = states

    def test(self):
        # initialize an empty dict for each metrics
        results = {}

        for metric in metrics:
            results[metric.__name__] = {}

        for state_name, state in states.items(): # loop through states
            for metric in metrics: # loop through metrics
                net.load_state_dict(state)
                for g, y in data: # loop through the dataset
                    


         



