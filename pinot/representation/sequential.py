""" Chain mutiple layers of GN together.
"""
import pinot
import torch
import dgl
import numpy as np

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
                activation = getattr(torch, exe)

                setattr(self, "a" + str(idx), activation)

                self.exes.append("a" + str(idx))

            # float -> dropout
            elif isinstance(exe, float):
                dropout = torch.nn.Dropout(exe)
                setattr(self, "o" + str(idx), dropout)

                self.exes.append("o" + str(idx))

    def forward(self, g, x=None, pool=lambda g: dgl.sum_nodes(g, "h")):

        if x is None:
            x = g.ndata["h"]

        x = self.f_in(x)

        for exe in self.exes:
            if exe.startswith("d"):
                x = getattr(self, exe)(g, x)
            else:
                x = getattr(self, exe)(x)

        if pool is not None:
            with g.local_scope():
                g.ndata["h"] = x
                x = pool(g)

        return x

    def pool(self, g, h, pool=lambda g: dgl.sum_nodes(g, "h")):
        with g.local_scope():
            g.ndata["h"] = h
            return pool(g)


class SequentialAttention(torch.nn.Module):
    """ Sequential module for Attention Graph Networks
    It follows very similar interface to Sequential. 

    Config should be a sequence of 
    
    [out_features, num_heads, pool_action, activation, out_features, ...]

    """
    def __init__(
        self,
        config,
        feature_units=117,
        input_units=128,
        model_kwargs={},
    ):
        super(SequentialAttention, self).__init__()

        # the initial dimensionality
        dim = input_units
        assert(len(config) % 4 == 0)
        layers = np.split(np.array(config, dtype=object), int(len(config)/4))

        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        # record the name of the layers in a list
        self.exes = []
        self.pool_layers = []
        prev_concat = False
        for idx, layer in enumerate(layers):
            out_features, num_heads, pool, activation = layer

            assert(pool in ["sum", "average", "concat"])

            # Initialize GATLayer
            gatLayer = dgl.nn.pytorch.GATConv(dim, out_features, num_heads, **model_kwargs)
            setattr(self, "d" + str(idx), gatLayer)
            dim = out_features if pool != "concat" else out_features * num_heads
            self.exes.append("d" + str(idx))
            
            # Activation layer
            activation = getattr(torch, activation)
            setattr(self, "a" + str(idx), activation)
            self.exes.append("a" + str(idx))

            # Pooling
            self.pool_layers.append(pool)

        # This module is initialized here and not used
        # 'Net' automatically determines the input dimension to output regressor by inspecting
        # the last out_features of 'Representation'. In the case of AttentionNetworks, the output
        # is 3D (due to multiple heads)
        self.dummy_output = torch.nn.Linear(1, out_features)

    def forward(self, g, h=None, pool=lambda g: dgl.sum_nodes(g, "h")):
        if h is None:
            h = g.ndata["h"]
        h = self.f_in(h)
        pool_id = 0

        with g.local_scope():
            for exe in self.exes:
                if exe.startswith("d"):
                    h = getattr(self, exe)(g, h)
                    if self.pool_layers[pool_id] == "concat":
                        h = h.view(h.shape[0], -1)
                    elif self.pool_layers[pool_id] == "average":
                        h = h.mean(1)
                    else:
                        h = h.sum(1)
                    pool_id += 1
                else:
                    h = getattr(self, exe)(h)

            if pool is not None:
                g.ndata["h"] = h
                h = pool(g)

            return h

    def pool(self, g, h, pool=lambda g: dgl.sum_nodes(g, "h")):
        with g.local_scope():
            g.ndata["h"] = h
            return pool(g)
