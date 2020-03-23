import pinot

def test_setup():
    gn = pinot.models.dgl_legacy
    nets = pinot.models.Sequential(gn, [128, 'tanh', 0.1])
