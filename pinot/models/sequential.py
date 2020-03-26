""" Chain mutiple layers of GN together.
"""
import pinot
import torch
import dgl

class Sequential(torch.nn.Module):
    def __init__(self, model, config, feature_units=117, input_units=128):
        super(Sequential, self).__init__()

        # the initial dimensionality
        dim = input_units

        # record the name of the layers in a list
        self.exes = []

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units),
            torch.nn.Tanh())

        # make a pytorch function on tensors on graphs
        def apply_atom_in_graph(fn):
            def _fn(g):
                g.apply_nodes(
                    lambda node: {'h': fn(node.data['h'])})
                return g
            return _fn

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except:
                pass

            # int -> feedfoward
            if type(exe) == int:
                setattr(
                    self,
                    'd' + str(idx),
                    model(dim, exe))

                dim = exe
                self.exes.append('d' + str(idx))

            # str -> activation
            elif type(exe) == str:
                activation = getattr(torch.nn.functional, exe)

                setattr(
                    self,
                    'a' + str(idx),
                    apply_atom_in_graph(activation))

                self.exes.append('a' + str(idx))

            # float -> dropout
            elif type(exe) == float:
                dropout = torch.nn.Dropout(exe)
                setattr(
                    self,
                    'o' + str(idx),
                    apply_atom_in_graph(dropout))

                self.exes.append('o' + str(idx))

        # readout
        self.f_out = torch.nn.Linear(dim, 1)


    def forward(self, g):

        g.apply_nodes(
            lambda nodes: {'h': self.f_in(nodes.data['h0'])})

        for exe in self.exes:
            g = getattr(self, exe)(g)

        y_hat = torch.squeeze(self.f_out(dgl.sum_nodes(g, 'h')))

        return y_hat
