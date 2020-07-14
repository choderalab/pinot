import pinot


def test_setup():
    """ """
    config = [
        "GraphConv",
        128,
        "activation",
        "tanh",
        "dropout",
        0.1,
        "GATConv",
        128,
        3,
        "attention_pool",
        "mean",
        "activation",
        "tanh",
    ]
    nets = pinot.representation.SequentialMix(config)
