""" Chain mutiple layers of GN together.
"""
import pinot
import torch
import dgl


class Sequential(torch.nn.Module):
    def __init__(
        self,
        layer,
        config,
        feature_units=117,
        input_units=128,
        model_kwargs={},
    ):
        super(Sequential, self).__init__()

        # the initial dimensionality
        dim = input_units

        # record the name of the layers in a list
        self.exes = []

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        # parse the config
        for idx, exe in enumerate(config):

            try:
                exe = float(exe)

                if exe >= 1:
                    exe = int(exe)
            except BaseException:
                pass

            # int -> feedfoward
            if isinstance(exe, int):
                setattr(self, "d" + str(idx), layer(dim, exe, **model_kwargs))

                dim = exe
                self.exes.append("d" + str(idx))

            # str -> activation
            elif isinstance(exe, str):
                activation = getattr(torch.nn.functional, exe)

                setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))

            # float -> dropout
            elif isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def forward(self, g, x=None, 
            pool=lambda g: dgl.sum_nodes(g, 'h')):
    
        if x is None:
            x = g.ndata['h']

        x = self.f_in(x)

        for exe in self.exes:
            if exe.startswith('d'):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        if pool is not None:
            g = g.local_var()
            g.ndata['h'] = x
            x = pool(g)

        return x
