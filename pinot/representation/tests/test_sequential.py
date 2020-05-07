import pinot

def test_setup():
    gn = pinot.representation.dgl_legacy.gn()
    nets = pinot.representation.Sequential(gn, [128, 'tanh', 0.1])
