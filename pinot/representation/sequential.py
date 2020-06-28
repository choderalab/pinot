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


# This dictionary is used to keep track of the parameters needed to
# initialize the various convolution layer as well as activation/dropout
# pooling
layer_param_type = {
    # Graph convolution layers
    "GraphConv": {"out_feats":int,},
    "GINConv":  {"out_feats":int,},
    "SAGEConv": {"out_feats":int,},
    "EdgeConv": {"out_feats":int,},
    "SAGEConv": {"out_feats":int,},
    "GATConv":  {"out_feats":int, "num_heads":int},
    "RelGraphConv": {"out_feats":int, "num_rel":int,},
    "TAGConv": {"out_feats":int, "k":int,},
    "SGConv": {"out_feats":int, "k":int,},
    "GatedGraphConv": {"out_feats": int, "n_steps": int, "n_etypes": int,},
    "ChebConv": {"out_feats":int, "k":int,},
    # Activation
    "activation": {"type":str},
    # Dropout
    "dropout": {"p":float},
    # Attention pooling
    "attention_pool": {"type":str}
}

def get_parameter(layer_config):
    """ Get the parameters needed to initialize a layer
    from the layer_config is which is only given as a list of strings

    Arg:
        layer_config: list[str]
            A list where the first element is the name of the layer
            type and the subsequent elements specify the parameters of the
            layer

    Return:
        parameters: dict
            A dictionary containing the parameters needed to initialize
            the layer. Each parameter has been converted from string to the
            appropriate data type
    """
    # Get the layer type
    layer_type = layer_config[0]
    # Get the config (exclude the layer type)
    config     = layer_config[1:]
    all_parameters = layer_param_type[layer_type]
    param_list = list(all_parameters.keys())
    parameters = {}
    # This assumes that layer_config lists the parameters in the same
    # order as listed above in layer_param_type
    # For example, the user needs to specify
    # "GATConv", 128, 3
    # where 128 is the output dimension and 3 is the number of attention
    # heads
    for i, param in enumerate(param_list):
        # If layer_config does not specify some of the parameters, stop
        # and leave those parameters as None
        if i == len(config):
            break
        # Set the parameters to the right type
        parameters[param] = all_parameters[param](config[i])
    return parameters

def initialize_layer(layer_config, in_feats):
    """ Initialize a layer given the layer configuration and the 
    specified input dimension

    Arg:
        layer_config: list[str]
            A list where the first element is the name of the layer
            type and the subsequent elements specify the parameters of the
            layer

        in_feats: int
            The specified input dimension (usually the output dimension)
            of the previous layer

    Returns:
        tuple(layer_type, layer, parameters)

        layer_type: str
            The name type of the layer

        layer: object
            The initialized layer

        parameters: dict
            A dictionary containing the parameters needed to initialize
            the layer. Each parameter has been converted from string to the
            appropriate data type
    """
    layer_type = layer_config[0]
    assert(layer_type in layer_param_type)
    # Get the parameters needed to initialize the layer
    parameters = get_parameter(layer_config)

    # If this is a Graph Convolution Layer
    if "out_feats" in parameters:
        out_feats  = parameters["out_feats"]
        extra_args = dict([(key, val) for (key,val) in parameters.items() if key != "out_feats"])
        layer_init = getattr(dgl.nn.pytorch.conv, layer_type)
        layer      = layer_init(in_feats, out_feats, **extra_args)

    else: # Activation or pooling or dropout
        if layer_type == "activation":
            layer = getattr(torch, parameters["type"])
        elif layer_type == "dropout":
            layer = torch.nn.Dropout(parameters["p"])
        else: # If "attention_pool"
            assert (parameters["type"] in ["concat", "mean", "sum"])
            layer = parameters["type"]

    return layer_type, layer, parameters

def get_config_layers(config):
    """ Parse the config from a list of strings into component layers configs,
    separated the keywords that are the layer type name

    Arg:
        config: list[str]
            A list specifying the layers and their configurations, given
            by user

    Returns:
        layers: list[list[str]]
            A list of list, each sub-list is the config for a layer
    """
    layers = []

    i = 0
    current_layer = []
    while i < len(config):
        # If config[i] specifies the layer type
        if (type(config[i])==str) and (config[i] in layer_param_type):
            layers.append(current_layer)
            current_layer = []
        current_layer.append(config[i])
        i += 1
    layers.append(current_layer)
    layers = layers[1:]
    return layers


class SequentialMix(torch.nn.Module):
    """ Module to allow for mixing of different convolution layer types
    """
    def __init__(
        self,
        config,
        feature_units=117,
        input_units=128,
        model_kwargs={},
    ):
        super(SequentialMix, self).__init__()

        # the initial dimensionality
        last_out_dim = input_units
        # initial featurization
        self.f_in = torch.nn.Sequential(
            torch.nn.Linear(feature_units, input_units), torch.nn.Tanh()
        )

        self.exes = []
        # Extract the invidual layers
        layers = get_config_layers(config)

        prev_layer_type   = None
        prev_layer_param  = None

        for idx, layer_config in enumerate(layers):
            layer_type, initialized_layer, layer_params = initialize_layer(layer_config, last_out_dim)

            if "out_feats" in layer_params:
                last_out_dim = layer_params["out_feats"]

            # If it's an activation/dropout or attention pooling layer
            if layer_type in ["activation", "dropout", "attention_pool"]:
                if layer_type == "activation":
                    setattr(self, "act" + str(idx), initialized_layer)
                    self.exes.append("act" + str(idx))
                elif layer_type == "dropout":
                    setattr(self, "drop" + str(idx), initialized_layer)
                    self.exes.append("drop" + str(idx))
                else: # Attention pooling layer
                    # We can only have attention pool after an attention layer
                    assert(prev_layer_type == "GATConv")
                    assert(type(initialized_layer) == str)
                    self.exes.append("attn_" + initialized_layer)

                    # If we're concatenating the attention heads, the output dimension gets
                    # multiplied by the number of heads, otherwise, it is the same as the 
                    # output dimension per head
                    last_out_dim = prev_layer_param["out_feats"] * prev_layer_param["num_heads"]\
                        if layer_params["type"] == "concat" else last_out_dim

            # If it's a graph convolution layer
            else:
                setattr(self, "conv" + str(idx), initialized_layer)
                self.exes.append("conv" + str(idx))
            
            # Keep track of the last layer
            prev_layer_type = layer_type
            prev_layer_param = layer_params

        # To explicitly state the output from the representation layer
        # This will ensure that the `Net` properly initializes a output
        # regression with the correct dimension
        self.dummy_output = torch.nn.Linear(1, last_out_dim)
        

    def forward(self, g, h=None, pool=lambda g: dgl.sum_nodes(g, "h")):
        if h is None:
            h = g.ndata["h"]
        h = self.f_in(h)

        with g.local_scope():
            for exe in self.exes:
                if exe.startswith("conv"):
                    h = getattr(self, exe)(g, h)
                # If attention pooling layer
                elif exe.startswith("attn_"):
                    assert(h.ndim == 3)
                    if exe[5:] == "concat":
                        h = h.view(h.shape[0], -1)
                    elif exe[5:] == "mean":
                        h = h.mean(1)
                    else:
                        h = h.sum(1)
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