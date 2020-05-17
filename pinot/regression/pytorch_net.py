import torch
import torch.nn as nn
import numpy as np
import pdb as pdb
import os as os

sigmoid = nn.Sigmoid()
tanh = nn.Tanh()
softplus = nn.Softplus()
relu=nn.ReLU()
elu=nn.ELU()

class Identity_layer(nn.Module):
    def __init__(self):
        super(Identity_layer, self).__init__()

    def forward(self, x):
        return x


def nonlinear_x(nonlinearity=None):
    if nonlinearity=='identity':
        return Identity_layer()
    if nonlinearity=='tanh':
        return nn.Tanh()
    if nonlinearity=='relu':
        return nn.ReLU()
    if nonlinearity=='elu':
        return nn.ELU()
    if nonlinearity=='softplus':
        return nn.Softplus()
    if nonlinearity=='sigmoid':
        return nn.Sigmoid()


class ListOutModule(nn.ModuleList):
    def __init__(self, modules):
        super(ListOutModule, self).__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm.forward(*args, **kwargs) for mm in self]


# example: regressor_net = NetClass(, n_input=graph_hidden_features, n_output=[1,1], hidden_dim=[10])
        
class NetClass(nn.Module):
    def __init__(self, n_input=None, n_output=[10,10], hidden_dim=[10,10], nonlinearity='softplus', dropout=False):
        super(NetClass, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.dropout = dropout
        num_layers = len(hidden_dim)
        layers=self.init_pre_layers(num_layers=num_layers, input_dim=n_input, hidden_dim=hidden_dim, nonlinearity=nonlinearity)
        num_outputs = len(n_output)
        out_layers=self.init_output_layers(num_outputs=num_outputs, input_dim=hidden_dim[-1], output_dim=n_output)
        self.mlp = self.init_layers(layers=layers, output_layers=out_layers)
        pass

    
    def init_pre_layers(self, num_layers=None, input_dim=None, hidden_dim=None, nonlinearity=None):
        layers = []
        s1 = input_dim
        #s2 = hidden_dim[0]
        for ix in range(num_layers):
            s2 = hidden_dim[ix]
            ll=nn.Linear(s1, s2)
            ll.weight.data = self.random_init_val(size=[s2, s1])
            ll.bias.data = self.random_init_val(size=[s2])
            layers.append(ll)
            #llbn=nn.BatchNorm1d(s2)
            #llbn=nn.InstanceNorm1d(s2)
            #layers.append(llbn)
            ll2 = nonlinear_x(nonlinearity)#nn.Softplus()
            layers.append(ll2)
            if self.dropout:
                m=nn.Dropout(p=0.5)
                layers.append(m)
            s1 = s2
            
        return layers

    def init_output_layers(self, num_outputs=3, input_dim=None, output_dim = [10,20,30]):
        output_lins = []
        for ix in range(num_outputs):
            ll=nn.Linear(input_dim, output_dim[ix])
            if 0:#ix > 0:
                ll.weight.data = self.random_init_val(size=[output_dim[ix], input_dim], offset=-2.)
                ll.bias.data = self.random_init_val(size=[output_dim[ix]], offset=-2.)
            else:
                ll.weight.data = self.random_init_val(size=[output_dim[ix], input_dim], offset=0.)#-2.)
                ll.bias.data = self.random_init_val(size=[output_dim[ix]], offset=0.)
            output_lins.append(ll)

        return output_lins


    def init_layers(self, layers=None, output_layers=None):
        layers.append(ListOutModule(output_layers))
        mlp = nn.Sequential(*layers)
        return mlp

    def random_init_val(self, size=None, init_std=0.02, offset=0.):
        """
        randomly initialize a neural network weight from a Gaussian with given variance
        """
        return (torch.randn(*size) * 2*init_std) + offset


    def forward(self, X=None):
        output = self.mlp.forward(X)
        return output


